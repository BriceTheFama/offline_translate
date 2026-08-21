import 'dart:typed_data';

import '../core/generation_config.dart';
import '../core/model_info.dart';
import '../exceptions/exceptions.dart';
import '../tokenizer/marian_tokenizer.dart';
import 'native/onnx_runtime_session.dart';
import 'native/onnx_runtime_tensor.dart';
import 'translation_engine.dart';

/// The greedy decoding loop, together with everything it reuses.
///
/// A runner owns run plans and scratch tensors, which are **not** safe to share
/// between isolates. The ONNX Runtime sessions underneath *are* safe to call
/// from several threads, so a worker isolate builds its own runner over the
/// same sessions rather than loading a second copy of the model. That is what
/// lets `translate()` move inference off the UI isolate while `translateSync()`
/// keeps running on the caller's.
///
/// Three properties drive its memory and speed:
///
/// * **A decoding step allocates nothing.** Input tensors, the two
///   `use_cache_branch` flags and the first-step placeholders are created once
///   here and rewritten in place afterwards. The only native objects created
///   per step are the outputs ONNX Runtime returns, and each is released before
///   the next step.
/// * **Names are encoded once.** A run plan owns its C strings for the life of
///   the session, instead of re-encoding 42 of them per generated token.
/// * **The KV cache never enters Dart.** Output tensor handles are bound
///   straight back as the next step's inputs.
class MarianRunner {
  MarianRunner._(
    this.model,
    this.tokenizer,
    this._encoder,
    this._decoder,
    this._maxInputTokens,
  );

  /// Metadata of the model being run.
  final ModelInfo model;

  /// Tokenizer for this model. One per runner, because Dart objects do not
  /// cross isolates.
  final MarianTokenizer tokenizer;

  final OrtSession _encoder;
  final OrtSession _decoder;
  final int _maxInputTokens;

  late final OrtRunPlan _encoderPlan;
  late final OrtRunPlan _firstStepPlan;
  late final OrtRunPlan _cachedStepPlan;

  late final OrtTensor _encoderIds;
  late final OrtTensor _encoderMask;
  late final OrtTensor _decoderInputIds;
  late final OrtTensor _useCacheFalse;
  late final OrtTensor _useCacheTrue;
  late final List<OrtTensor> _emptyPast;
  late final Int64List _encoderIdsView;
  late final Int64List _encoderMaskView;
  late final Int64List _decoderInputView;
  late final List<OrtTensor?> _encoderOutputs;
  late final List<OrtTensor?> _firstStepOutputs;
  late final List<OrtTensor?> _cachedStepOutputs;

  late final List<String> _pastDecoderNames;
  late final List<String> _pastEncoderNames;
  late final int _kvPerStep;

  bool _disposed = false;

  static const String _inputIdsName = 'input_ids';
  static const String _encoderMaskName = 'encoder_attention_mask';
  static const String _encoderStatesName = 'encoder_hidden_states';
  static const String _useCacheBranchName = 'use_cache_branch';

  /// Builds the plans and scratch tensors for [encoder] and [decoder].
  static MarianRunner create({
    required ModelInfo model,
    required MarianTokenizer tokenizer,
    required OrtSession encoder,
    required OrtSession decoder,
    required int maxInputTokens,
  }) {
    if (!decoder.outputNames.contains('next_token')) {
      throw const TranslationEngineException(
          'decoder.onnx has no `next_token` output; the model bundle was not '
          'produced by tool/build_model.py');
    }
    final runner =
        MarianRunner._(model, tokenizer, encoder, decoder, maxInputTokens);
    runner._build();
    return runner;
  }

  void _build() {
    final arch = model.architecture;
    final layers = arch.decoderLayers;
    _kvPerStep = layers * 2;

    _pastDecoderNames = <String>[
      for (var l = 0; l < layers; l++) ...<String>[
        'past_key_values.$l.decoder.key',
        'past_key_values.$l.decoder.value',
      ],
    ];
    _pastEncoderNames = <String>[
      for (var l = 0; l < layers; l++) ...<String>[
        'past_key_values.$l.encoder.key',
        'past_key_values.$l.encoder.value',
      ],
    ];
    final presentDecoder = <String>[
      for (var l = 0; l < layers; l++) ...<String>[
        'present.$l.decoder.key',
        'present.$l.decoder.value',
      ],
    ];
    final presentEncoder = <String>[
      for (var l = 0; l < layers; l++) ...<String>[
        'present.$l.encoder.key',
        'present.$l.encoder.value',
      ],
    ];

    _encoderPlan = _encoder.plan(
      const <String>['input_ids', 'attention_mask'],
      const <String>['last_hidden_state'],
    );
    final decoderInputs = <String>[
      _inputIdsName,
      _encoderMaskName,
      _encoderStatesName,
      ..._pastDecoderNames,
      ..._pastEncoderNames,
      _useCacheBranchName,
    ];
    _firstStepPlan = _decoder.plan(
      decoderInputs,
      <String>['next_token', ...presentDecoder, ...presentEncoder],
    );
    _cachedStepPlan = _decoder.plan(
      decoderInputs,
      <String>['next_token', ...presentDecoder],
    );

    _encoderIds = OrtTensor.int64(<int>[1, _maxInputTokens]);
    _encoderMask = OrtTensor.int64(<int>[1, _maxInputTokens]);
    _encoderIdsView = _encoderIds.asInt64List(_maxInputTokens);
    _encoderMaskView = _encoderMask.asInt64List(_maxInputTokens);
    _decoderInputIds = OrtTensor.int64(<int>[1, 1]);
    _decoderInputView = _decoderInputIds.asInt64List(1);
    _useCacheFalse = OrtTensor.boolean(<int>[1], value: false);
    _useCacheTrue = OrtTensor.boolean(<int>[1], value: true);
    _emptyPast = <OrtTensor>[
      for (var i = 0; i < _kvPerStep * 2; i++)
        OrtTensor.float32(
            <int>[1, arch.decoderAttentionHeads, 0, arch.headDimension]),
    ];

    _encoderOutputs = List<OrtTensor?>.filled(1, null);
    _firstStepOutputs = List<OrtTensor?>.filled(1 + _kvPerStep * 2, null);
    _cachedStepOutputs = List<OrtTensor?>.filled(1 + _kvPerStep, null);
  }

  /// Runs greedy generation for [inputIds] on the calling thread.
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

    OrtTensor? hidden;
    final pastDecoder = List<OrtTensor?>.filled(_kvPerStep, null);
    final pastEncoder = List<OrtTensor?>.filled(_kvPerStep, null);
    final tokens = <int>[];
    var truncated = false;
    var encodeMicros = 0;

    try {
      _encoderPlan
        ..setInputAt(0, ids)
        ..setInputAt(1, mask)
        ..run(_encoderOutputs);
      hidden = _encoderOutputs[0];
      encodeMicros = watch.elapsedMicroseconds;

      _firstStepPlan
        ..setInput(_inputIdsName, _decoderInputIds)
        ..setInput(_encoderMaskName, mask)
        ..setInput(_encoderStatesName, hidden!)
        ..setInput(_useCacheBranchName, _useCacheFalse);
      _cachedStepPlan
        ..setInput(_inputIdsName, _decoderInputIds)
        ..setInput(_encoderMaskName, mask)
        ..setInput(_encoderStatesName, hidden)
        ..setInput(_useCacheBranchName, _useCacheTrue);
      for (var i = 0; i < _kvPerStep; i++) {
        _firstStepPlan
          ..setInput(_pastDecoderNames[i], _emptyPast[i])
          ..setInput(_pastEncoderNames[i], _emptyPast[_kvPerStep + i]);
      }

      final maxNew = _effectiveMaxNewTokens(sourceLength, config);
      _decoderInputView[0] = arch.decoderStartTokenId;
      final decodeWatch = Stopwatch()..start();

      for (var step = 0; step < maxNew; step++) {
        final first = step == 0;
        final outputs = first ? _firstStepOutputs : _cachedStepOutputs;
        (first ? _firstStepPlan : _cachedStepPlan).run(outputs);

        final next = outputs[0]!.int64At(0);
        outputs[0]!.release();

        for (var i = 0; i < _kvPerStep; i++) {
          pastDecoder[i]?.release();
          pastDecoder[i] = outputs[1 + i];
          _cachedStepPlan.setInput(_pastDecoderNames[i], pastDecoder[i]!);
        }
        if (first) {
          for (var i = 0; i < _kvPerStep; i++) {
            pastEncoder[i] = outputs[1 + _kvPerStep + i];
            _cachedStepPlan.setInput(_pastEncoderNames[i], pastEncoder[i]!);
          }
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
      for (final tensor in pastDecoder) {
        tensor?.release();
      }
      for (final tensor in pastEncoder) {
        tensor?.release();
      }
      hidden?.release();
      ids.release();
      mask.release();
      _encoderPlan.clearInputs();
      _firstStepPlan.clearInputs();
      _cachedStepPlan.clearInputs();
    }
  }

  int _effectiveMaxNewTokens(int sourceLength, GenerationConfig config) {
    final byRatio = (sourceLength * config.lengthRatioLimit).ceil() + 8;
    final limit = byRatio < config.maxNewTokens ? byRatio : config.maxNewTokens;
    final positions = model.architecture.maxPositionEmbeddings;
    return limit < positions ? limit : positions;
  }

  /// Translates [text] end to end: tokenize, generate, detokenize.
  ({String text, bool truncated}) translate(
      String text, GenerationConfig config) {
    final output = generate(tokenizer.encode(text), config);
    return (text: tokenizer.decode(output.tokens), truncated: output.truncated);
  }

  /// Releases the scratch tensors owned by this runner. The sessions are left
  /// alone; whoever created them releases them.
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    for (final tensor in _emptyPast) {
      tensor.release();
    }
    _encoderIds.release();
    _encoderMask.release();
    _decoderInputIds.release();
    _useCacheFalse.release();
    _useCacheTrue.release();
  }
}
