import 'dart:typed_data';

import '../core/generation_config.dart';
import '../core/model_info.dart';
import '../exceptions/exceptions.dart';
import '../tokenizer/marian_tokenizer.dart';
import 'marian_runner.dart';
import 'native/onnx_runtime_session.dart';
import 'ssru_runner.dart';
import 'translation_engine.dart';

/// The greedy decoding loop for one model family.
///
/// A runner owns run plans and scratch tensors, which are **not** safe to share
/// between isolates. The ONNX Runtime sessions underneath *are* safe to call
/// from several threads, so a worker isolate builds its own runner over the
/// same sessions rather than loading a second copy of the model. That is what
/// lets `translateLong()` move inference off the UI isolate while `translate()`
/// keeps running on the caller's.
///
/// There are two implementations because the two model families need genuinely
/// different loops, not because the code was factored for its own sake:
///
/// * [MarianRunner] drives optimum's merged decoder — a growing key/value cache
///   per layer, and an `If` node selecting a first-step or cached-step branch.
/// * [SsruRunner] drives the Firefox Translations students, whose decoder
///   self-attention is a recurrent unit. Their entire history is one small
///   state per layer, so there is no cache to grow and no branch to select.
abstract class DecodingRunner {
  /// Metadata of the model being run.
  ModelInfo get model;

  /// Tokenizer for this model. One per runner, because Dart objects do not
  /// cross isolates.
  MarianTokenizer get tokenizer;

  /// Runs greedy generation for [inputIds] on the calling thread.
  GenerationOutput generate(Int32List inputIds, GenerationConfig config);

  /// Translates [text] end to end: tokenize, generate, detokenize.
  ({String text, bool truncated}) translate(
      String text, GenerationConfig config);

  /// Releases the scratch tensors owned by this runner. The sessions are left
  /// alone; whoever created them releases them.
  void dispose();
}

/// Builds the runner the [model]'s architecture calls for.
DecodingRunner createRunner({
  required ModelInfo model,
  required MarianTokenizer tokenizer,
  required OrtSession encoder,
  required OrtSession decoder,
  required int maxInputTokens,
}) {
  switch (model.architecture.family) {
    case ModelFamily.marian:
      return MarianRunner.create(
        model: model,
        tokenizer: tokenizer,
        encoder: encoder,
        decoder: decoder,
        maxInputTokens: maxInputTokens,
      );
    case ModelFamily.tinySsru:
      return SsruRunner.create(
        model: model,
        tokenizer: tokenizer,
        encoder: encoder,
        decoder: decoder,
        maxInputTokens: maxInputTokens,
      );
  }
}

/// Shared length policy: never generate more than the model can position, nor
/// more than the caller allows, nor wildly more than the input length.
int effectiveMaxNewTokens(
    ModelInfo model, int sourceLength, GenerationConfig config) {
  final byRatio = (sourceLength * config.lengthRatioLimit).ceil() + 8;
  final limit = byRatio < config.maxNewTokens ? byRatio : config.maxNewTokens;
  final positions = model.architecture.maxPositionEmbeddings;
  return limit < positions ? limit : positions;
}

/// Thrown when a bundle's graphs do not match the family its manifest declares.
TranslationEngineException wrongGraph(ModelInfo model, String detail) =>
    TranslationEngineException(
        'Model "${model.id}" declares family ${model.architecture.family.name} '
        'but $detail; the bundle was not produced by this package\'s tooling');
