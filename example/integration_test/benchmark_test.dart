import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:offline_translate/src/engine/onnx_engine.dart' show OnnxEngine;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Reported alongside the numbers so a table can never be misattributed to the
/// wrong runtime.
String runtimeVersion = 'unknown';

/// Benchmark harness. Prints a table of cold start, warm inference and memory
/// figures for the current device.
///
/// ```sh
/// flutter test integration_test/benchmark_test.dart -d macos \
///   --dart-define=OT_MODELS_DIR=$HOME/ot-models
/// ```
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const localDir = String.fromEnvironment('OT_MODELS_DIR', defaultValue: '');
  const remoteUrl = String.fromEnvironment('OT_MODELS_URL', defaultValue: '');

  const short = 'Hello world';
  const sentence =
      'The committee agreed that the new proposal should be '
      'reviewed carefully before any final decision is taken next month.';

  String words(int count) {
    const sample =
        'The quick brown fox jumps over the lazy dog near the '
        'river bank while the sun slowly sets behind the distant hills and '
        'the evening air turns cold. ';
    final buffer = StringBuffer();
    var produced = 0;
    while (produced < count) {
      buffer.write(sample);
      produced += 26;
    }
    return buffer.toString().trim();
  }

  String paragraphs(int count) {
    final buffer = StringBuffer();
    for (var i = 0; i < count; i++) {
      buffer
        ..write(words(60))
        ..write('\n\n');
    }
    return buffer.toString().trim();
  }

  late Directory workDir;
  late FileModelManager manager;
  final rows = <List<String>>[];

  void record(String label, Object value, [String? extra]) {
    rows.add(<String>[label, '$value', extra ?? '']);
  }

  int rssMb() => (ProcessInfo.currentRss / 1048576).round();

  setUpAll(() async {
    final support = await getApplicationSupportDirectory();
    workDir = Directory(p.join(support.path, 'ot-benchmark'));
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
    workDir.createSync(recursive: true);

    ModelSource source;
    if (localDir.isNotEmpty && Directory(localDir).existsSync()) {
      source = DirectoryModelSource(localDir);
    } else if (remoteUrl.isNotEmpty) {
      source = HttpModelSource(baseUrl: Uri.parse(remoteUrl));
    } else {
      final docs = await getApplicationDocumentsDirectory();
      final pushed = Directory(p.join(docs.path, 'ot-models'));
      // Zero configuration: fall back to the published bundles, so this runs
      // on a fresh device with nothing but a network connection.
      source = pushed.existsSync()
          ? DirectoryModelSource(pushed.path)
          : HttpModelSource.official();
    }
    manager = FileModelManager(source: source, rootPath: workDir.path);
  });

  tearDownAll(() {
    // ignore: avoid_print
    print(
      '\n=== offline_translate benchmark ==='
      '\nplatform: ${Platform.operatingSystem} '
      '${Platform.operatingSystemVersion}'
      '\ncores: ${Platform.numberOfProcessors}\n',
    );
    final width = rows.fold<int>(
      0,
      (a, r) => r[0].length > a ? r[0].length : a,
    );
    for (final row in rows) {
      // ignore: avoid_print
      print('${row[0].padRight(width)}  ${row[1].padLeft(10)}  ${row[2]}');
    }
    // ignore: avoid_print
    print('=== end benchmark ===\n');
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
  });

  testWidgets('cold start and warm inference', (tester) async {
    final rssBefore = rssMb();
    runtimeVersion = OnnxEngine.runtimeVersion;

    final installWatch = Stopwatch()..start();
    final info = await manager.install(
      const LanguagePair(Language.en, Language.fr),
    );
    installWatch.stop();
    record(
      'model size',
      '${(info.size / 1048576).toStringAsFixed(1)} MB',
      info.quantization,
    );
    record('install (verify + copy)', '${installWatch.elapsedMilliseconds} ms');

    // Cold start: engine construction, ONNX session creation, tokenizer parse.
    final coldWatch = Stopwatch()..start();
    final translator = await OfflineTranslator.initialize(
      languages: const {Language.en, Language.fr},
      defaultLanguage: Language.fr,
      modelManager: manager,
    );
    coldWatch.stop();
    record('cold start (load model)', '${coldWatch.elapsedMilliseconds} ms');
    record('RSS after load', '${rssMb() - rssBefore} MB', 'delta');
    record('RSS total', '${rssMb()} MB');

    // First inference on a freshly loaded model pays ONNX Runtime's one-off
    // arena allocation, so it is reported separately from the warm figures.
    final firstWatch = Stopwatch()..start();
    translator.translate(short, from: Language.en, to: Language.fr);
    firstWatch.stop();
    record('first translation', '${firstWatch.elapsedMilliseconds} ms', short);

    for (final entry in <(String, String)>[
      ('"Hello world"', short),
      ('20-word sentence', sentence),
      ('100 words', words(100)),
      ('500 words', words(500)),
    ]) {
      final text = entry.$2;
      // Three runs, keep the median, cache disabled by construction.
      final samples = <int>[];
      late TranslationResult result;
      for (var i = 0; i < 3; i++) {
        final watch = Stopwatch()..start();
        result = translator.translate(text, from: Language.en, to: Language.fr);
        samples.add(watch.elapsedMicroseconds);
      }
      samples.sort();
      final median = samples[1] / 1000;
      record(
        'warm ${entry.$1}',
        '${median.toStringAsFixed(1)} ms',
        '${result.chunkCount} chunk(s), '
            '${result.translatedText.length} chars out',
      );
    }

    // Long document through the async API.
    final document = paragraphs(20);
    final docWatch = Stopwatch()..start();
    final docResult = await translator.translateLong(
      document,
      from: Language.en,
      to: Language.fr,
    );
    docWatch.stop();
    record(
      'async 20-paragraph doc',
      '${docWatch.elapsedMilliseconds} ms',
      '${document.length} chars in, ${docResult.chunkCount} chunks',
    );
    record('RSS after long doc', '${rssMb()} MB');

    await translator.dispose();
    record('RSS after dispose', '${rssMb()} MB');
  }, timeout: const Timeout(Duration(minutes: 20)));

  testWidgets('cache hit cost', (tester) async {
    final translator = await OfflineTranslator.initialize(
      languages: const {Language.en, Language.fr},
      defaultLanguage: Language.fr,
      modelManager: manager,
      cache: TranslationCache(maxEntries: 64),
    );
    translator.translate(sentence, from: Language.en, to: Language.fr);
    final watch = Stopwatch()..start();
    final hit = translator.translate(
      sentence,
      from: Language.en,
      to: Language.fr,
    );
    watch.stop();
    expect(hit.fromCache, isTrue);
    record('cache hit', '${watch.elapsedMicroseconds} µs');
    await translator.dispose();
  }, timeout: const Timeout(Duration(minutes: 10)));
}
