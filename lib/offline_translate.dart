/// 100% offline neural machine translation for Flutter.
///
/// Runs OPUS-MT (MarianMT) encoder-decoder models locally with ONNX Runtime.
/// After a model is installed, translation performs no network access at all.
///
/// See `OfflineTranslator` for the entry point.
library;

export 'src/cache/translation_cache.dart' show TranslationCache;
export 'src/core/generation_config.dart' show GenerationConfig;
export 'src/core/language.dart' show Language, LanguagePair;
export 'src/core/runtime_config.dart'
    show Accelerator, GraphOptimization, RuntimeConfig;
export 'src/core/model_info.dart' show ModelArchitecture, ModelFile, ModelInfo;
export 'src/core/offline_translator.dart' show EngineFactory, OfflineTranslator;
export 'src/core/translation_result.dart' show TranslationResult;
export 'src/engine/translation_engine.dart'
    show ChunkTranslation, GenerationOutput, TranslationEngine;
export 'src/exceptions/exceptions.dart';
export 'src/model_manager/directory_model_source.dart'
    show DirectoryModelSource;
export 'src/model_manager/http_model_source.dart' show HttpModelSource;
export 'src/model_manager/model_manager.dart'
    show FileModelManager, ModelManager;
export 'src/model_manager/model_source.dart'
    show InstallProgress, InstallStage, ModelSource;
export 'src/utils/text_segmenter.dart' show TextChunk, TextSegmenter;
