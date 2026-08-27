import 'dart:typed_data';

import '../core/generation_config.dart';
import '../core/model_info.dart';
import '../exceptions/exceptions.dart';
import '../tokenizer/marian_tokenizer.dart';
import 'decoding_runner.dart';
import 'native/onnx_runtime_session.dart';
import 'native/onnx_runtime_tensor.dart';
import 'translation_engine.dart';

/// Greedy decoding for the Firefox Translations student models.
///
/// The graphs this drives are shaped by one architectural fact: the decoder's
/// self-attention is a Simpler Simple Recurrent Unit, not attention. Three
/// things follow, and all three make the loop simpler than [MarianRunner]'s.
///
/// * **There is no key/value cache.** The decoder's entire history is one
///   `[1, 1, d_model]` state per layer — 6 KB in total for a 4-layer model,
///   whatever the output length. Memory per decoding step is constant instead
///   of growing, and a long document costs no more per token than a short one.
/// * **There is no `use_cache_branch`.** A zero state *is* the first step, so
///   one graph and one run plan serve every step. `MarianRunner` needs two
///   plans and an `If` node to switch between them.
/// * **Cross-attention keys and values are computed in the encoder graph.**
///   They depend only on the source, so the decoder reads them from an input
///   and the encoder's hidden states never cross the FFI boundary at all.
///
/// The step-0 convention comes from Marian: decoding starts from an all-zero
/// embedding rather than a start-of-sequence token. The bundle records that as
/// `decoder_start_token_id: -1`, and the graph turns any negative id into a
/// zero embedding.
///
/// Like [MarianRunner], a decoding step allocates nothing: the input tensors
/// are created once and rewritten in place, the state tensors ONNX Runtime
/// returns are bound straight back as the next step's inputs without entering
/// Dart, and only the 8-byte `next_token` is ever read.
class SsruRunner implements DecodingRunner {
  SsruRunner._(
    this.model,
    this.tokenizer,
    this._encoder,
    this._decoder,
    this._maxInputTokens,
  );

  @override
  final ModelInfo model;

  @override
  final MarianTokenizer tokenizer;

  final OrtSession _encoder;
  final OrtSession _decoder;
  final int _maxInputTokens;

  late final OrtRunPlan _encoderPlan;
  late final OrtRunPlan _decoderPlan;

  late final OrtTensor _encoderIds;
  late final OrtTensor _encoderMask;
  late final OrtTensor _decoderInputIds;
  late final OrtTensor _position;
  late final List<OrtTensor> _zeroStates;
  late final Int64List _encoderIdsView;
  late final Int64List _encoderMaskView;
  late final Int64List _decoderInputView;
  late final Int64List _positionView;
  late final List<OrtTensor?> _encoderOutputs;
  late final List<OrtTensor?> _decoderOutputs;

  late final List<String> _crossNames;
  late final List<String> _stateNames;
  late final int _layers;

  bool _disposed = false;

  static const String _inputIdsName = 'input_ids';
  static const String _positionName = 'position';
  static const String _encoderMaskName = 'encoder_attention_mask';

  /// Builds the plans and scratch tensors for [encoder] and [decoder].
  static SsruRunner create({
    required ModelInfo model,
    required MarianTokenizer tokenizer,
    required OrtSession encoder,
    required OrtSession decoder,
    required int maxInputTokens,
  }) {
    if (!decoder.outputNames.contains('next_token')) {
      throw wrongGraph(model, 'decoder.onnx has no `next_token` output');
    }
    if (!decoder.inputNames.contains(_positionName)) {
      throw wrongGraph(model, 'decoder.onnx has no `position` input');
    }
    final runner =
        SsruRunner._(model, tokenizer, encoder, decoder, maxInputTokens);
    runner._build();
    return runner;
  }

  void _build() {
    final arch = model.architecture;
    _layers = arch.decoderLayers;
    final dimension = arch.decoderAttentionHeads * arch.headDimension;

    // The encoder emits the cross-attention key and value of every decoder
    // layer, interleaved, which is the order `tool/build_tiny_model.py` names
    // its outputs in.
    _crossNames = <String>[
      for (var l = 0; l < _layers; l++) ...<String>[
        'cross_key.$l',
        'cross_value.$l',
      ],
    ];
    _stateNames = <String>[for (var l = 0; l < _layers; l++) 'state.$l'];
    final newStateNames = <String>[
      for (var l = 0; l < _layers; l++) 'new_state.$l',
    ];

    _encoderPlan = _encoder.plan(
      const <String>['input_ids', 'attention_mask'],
      _crossNames,
    );
    _decoderPlan = _decoder.plan(
      <String>[
        _inputIdsName,
        _positionName,
        _encoderMaskName,
        ..._crossNames,
        ..._stateNames,
      ],
      <String>['next_token', ...newStateNames],
    );

    _encoderIds = OrtTensor.int64(<int>[1, _maxInputTokens]);
    _encoderMask = OrtTensor.int64(<int>[1, _maxInputTokens]);
    _encoderIdsView = _encoderIds.asInt64List(_maxInputTokens);
    _encoderMaskView = _encoderMask.asInt64List(_maxInputTokens);
    _decoderInputIds = OrtTensor.int64(<int>[1, 1]);
    _decoderInputView = _decoderInputIds.asInt64List(1);
    _position = OrtTensor.int64(<int>[1]);
    _positionView = _position.asInt64List(1);
    // `calloc` zeroes these, and nothing ever writes to them: they are the
    // step-0 state, which is what "no history yet" means for an SSRU.
    _zeroStates = <OrtTensor>[
      for (var l = 0; l < _layers; l++)
        OrtTensor.float32(<int>[1, 1, dimension]),
    ];

    _encoderOutputs = List<OrtTensor?>.filled(_crossNames.length, null);
    _decoderOutputs = List<OrtTensor?>.filled(1 + _layers, null);
  }

  @override
  GenerationOutput generate(Int32List inputIds, GenerationConfig config) {
    if (_disposed) {
      throw const TranslationEngineException('Runner has been disposed');
    }
    if (inputIds.isEmpty) {
      return const GenerationOutput(
          tokens: <int>[], truncated: false, encodeMicros: 0, decodeMicros: 0);
    }

    final arch = model.architecture;
    final sourceLength =
        inputIds.length > _maxInputTokens ? _maxInputTokens : inputIds.length;
    final watch = Stopwatch()..start();

    for (var i = 0; i < sourceLength; i++) {
      _encoderIdsView[i] = inputIds[i];
      _encoderMaskView[i] = 1;
    }
    // The encoder buffers are allocated at their maximum width once; a shorter
    // input just needs another OrtValue over the same memory, which is a
    // pointer wrap rather than a copy.
    final shape = <int>[1, sourceLength];
    final ids = OrtTensor.viewOf(_encoderIds, shape, bytesPerElement: 8);
    final mask = OrtTensor.viewOf(_encoderMask, shape, bytesPerElement: 8);

    final cross = List<OrtTensor?>.filled(_crossNames.length, null);
    final states = List<OrtTensor?>.filled(_layers, null);
    final tokens = <int>[];
    var truncated = false;
    var encodeMicros = 0;

    try {
      _encoderPlan
        ..setInputAt(0, ids)
        ..setInputAt(1, mask)
        ..run(_encoderOutputs);
      for (var i = 0; i < cross.length; i++) {
        cross[i] = _encoderOutputs[i];
      }
      encodeMicros = watch.elapsedMicroseconds;

      _decoderPlan
        ..setInput(_inputIdsName, _decoderInputIds)
        ..setInput(_positionName, _position)
        ..setInput(_encoderMaskName, mask);
      for (var i = 0; i < cross.length; i++) {
        _decoderPlan.setInput(_crossNames[i], cross[i]!);
      }
      for (var l = 0; l < _layers; l++) {
        _decoderPlan.setInput(_stateNames[l], _zeroStates[l]);
      }

      final maxNew = effectiveMaxNewTokens(model, sourceLength, config);
      _decoderInputView[0] = arch.decoderStartTokenId;
      final decodeWatch = Stopwatch()..start();

      for (var step = 0; step < maxNew; step++) {
        _positionView[0] = step;
        _decoderPlan.run(_decoderOutputs);

        final next = _decoderOutputs[0]!.int64At(0);
        _decoderOutputs[0]!.release();

        for (var l = 0; l < _layers; l++) {
          states[l]?.release();
          states[l] = _decoderOutputs[1 + l];
          _decoderPlan.setInput(_stateNames[l], states[l]!);
        }

        if (next == arch.eosTokenId) break;
        tokens.add(next);
        _decoderInputView[0] = next;
        if (step == maxNew - 1) truncated = true;
      }

      return GenerationOutput(
        tokens: tokens,
        truncated: truncated,
        encodeMicros: encodeMicros,
        decodeMicros: decodeWatch.elapsedMicroseconds,
      );
    } catch (e) {
      if (e is OfflineTranslatorException) rethrow;
      throw TranslationEngineException('Generation failed for ${model.id}', e);
    } finally {
      // Everything ONNX Runtime handed back during this call goes away here, so
      // repeated translations do not accumulate native memory. The plans then
      // forget those pointers, because they are dangling from now on.
      for (final tensor in states) {
        tensor?.release();
      }
      for (final tensor in cross) {
        tensor?.release();
      }
      ids.release();
      mask.release();
      _encoderPlan.clearInputs();
      _decoderPlan.clearInputs();
    }
  }

  @override
  ({String text, bool truncated}) translate(
      String text, GenerationConfig config) {
    final output = generate(tokenizer.encode(text), config);
    return (text: tokenizer.decode(output.tokens), truncated: output.truncated);
  }

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    for (final tensor in _zeroStates) {
      tensor.release();
    }
    _encoderIds.release();
    _encoderMask.release();
    _decoderInputIds.release();
    _position.release();
  }
}
