import 'package:meta/meta.dart';

import 'language.dart';

/// The outcome of a translation request.
@immutable
class TranslationResult {
  /// Creates a translation result.
  const TranslationResult({
    required this.sourceText,
    required this.translatedText,
    required this.sourceLanguage,
    required this.targetLanguage,
    required this.duration,
    this.fromCache = false,
    this.chunkCount = 1,
    this.truncated = false,
  });

  /// The text that was submitted.
  final String sourceText;

  /// The translated text.
  final String translatedText;

  /// Language the text was translated from.
  final Language sourceLanguage;

  /// Language the text was translated into.
  final Language targetLanguage;

  /// Wall-clock time spent producing [translatedText].
  final Duration duration;

  /// Whether the result was served from the translation cache.
  final bool fromCache;

  /// How many chunks the source text was split into.
  final int chunkCount;

  /// Whether generation stopped on the length limit instead of the end-of-
  /// sequence token, meaning the output may be cut short.
  final bool truncated;

  /// Language pair this result belongs to.
  LanguagePair get pair => LanguagePair(sourceLanguage, targetLanguage);

  @override
  String toString() => 'TranslationResult(${pair.id}, '
      '${duration.inMilliseconds}ms, chunks: $chunkCount'
      '${fromCache ? ', cached' : ''}${truncated ? ', truncated' : ''})';
}
