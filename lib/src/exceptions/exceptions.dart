import '../core/language.dart';

/// Base class for every error thrown by `offline_translate`.
abstract class OfflineTranslatorException implements Exception {
  /// Creates an exception carrying a human readable [message].
  const OfflineTranslatorException(this.message);

  /// Human readable description of what went wrong.
  final String message;

  @override
  String toString() => '$runtimeType: $message';
}

/// Thrown when a translation is requested for a model that is not installed.
class ModelNotInstalledException extends OfflineTranslatorException {
  /// Creates the exception for the model identified by [modelId].
  const ModelNotInstalledException(this.modelId)
      : super('Model "$modelId" is not installed. '
            'Call installModel() first (requires network access once).');

  /// Identifier of the missing model, e.g. `en-fr`.
  final String modelId;
}

/// Thrown when [translateSync] is called before the model is in memory.
///
/// The synchronous API deliberately refuses to block on disk I/O, so the model
/// has to be loaded first with `preload()`, by passing `from:`/`to:` to
/// `OfflineTranslator.initialize`, or by awaiting one `translate()` call.
class ModelNotLoadedException extends OfflineTranslatorException {
  /// Creates the exception for the model identified by [modelId].
  const ModelNotLoadedException(this.modelId)
      : super('Model "$modelId" is not loaded. translateSync() needs the model '
            'in memory: call preload(from:, to:) first, initialize the '
            'translator with that direction, or await translate() once.');

  /// Identifier of the model that is not resident, e.g. `en-fr`.
  final String modelId;
}

/// Thrown when a language direction has no model in the catalogue.
class UnsupportedLanguagePairException extends OfflineTranslatorException {
  /// Creates the exception for the unsupported [modelId].
  const UnsupportedLanguagePairException(this.modelId)
      : super('No model is available for direction "$modelId".');

  /// Identifier of the unsupported direction, e.g. `fr-de`.
  final String modelId;
}

/// Thrown when a translation names a language the translator was not
/// initialized with.
///
/// `OfflineTranslator.initialize(languages: ...)` declares the set an
/// application needs, and asking for anything outside it is a programming
/// error rather than a missing download — the point of declaring the set is
/// that models for the rest are never required.
class UnsupportedLanguageException extends OfflineTranslatorException {
  /// Creates the exception for [language], listing what was declared.
  UnsupportedLanguageException(this.language, this.declared)
      : super('Language "${language.code}" was not declared at initialize(); '
            'this translator serves '
            '${declared.map((l) => l.code).join(", ")}.');

  /// The language that was asked for.
  final Language language;

  /// The languages this translator was initialized with.
  final Set<Language> declared;
}

/// Thrown when installed model files fail their integrity check.
class ModelCorruptedException extends OfflineTranslatorException {
  /// Creates the exception, describing which [file] failed and why.
  const ModelCorruptedException(this.modelId, this.file, String reason)
      : super('Model "$modelId" is corrupted ($file): $reason');

  /// Identifier of the corrupted model.
  final String modelId;

  /// The file that failed verification.
  final String file;
}

/// Thrown when a model download fails.
class ModelDownloadException extends OfflineTranslatorException {
  /// Creates the exception for [modelId] with a description of the failure.
  const ModelDownloadException(this.modelId, String reason)
      : super('Failed to download model "$modelId": $reason');

  /// Identifier of the model that could not be downloaded.
  final String modelId;
}

/// Thrown when there is not enough free disk space to install a model.
class InsufficientStorageException extends OfflineTranslatorException {
  /// Creates the exception given [requiredBytes] and [availableBytes].
  const InsufficientStorageException(this.requiredBytes, this.availableBytes)
      : super('Not enough free space: $requiredBytes bytes required, '
            '$availableBytes bytes available.');

  /// Number of bytes the installation needs.
  final int requiredBytes;

  /// Number of bytes currently free on the target volume.
  final int availableBytes;
}

/// Thrown when the inference engine fails.
class TranslationEngineException extends OfflineTranslatorException {
  /// Creates the exception with a [message] and the optional [cause].
  const TranslationEngineException(super.message, [this.cause]);

  /// The underlying error, when the failure originated in ONNX Runtime.
  final Object? cause;

  @override
  String toString() => cause == null
      ? super.toString()
      : '${super.toString()} — caused by: $cause';
}

/// Thrown when the translator (or one of its models) has been disposed.
class TranslatorDisposedException extends OfflineTranslatorException {
  /// Creates the exception.
  const TranslatorDisposedException()
      : super('This OfflineTranslator has been disposed.');
}
