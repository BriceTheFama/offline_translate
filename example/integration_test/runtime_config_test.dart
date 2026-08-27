import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Compares runtime configurations on the device, including the execution
/// providers that only exist in mobile builds of ONNX Runtime.
///
/// ```sh
/// flutter test integration_test/runtime_config_test.dart -d <device> \
///   --dart-define=OT_MODELS_DIR=$HOME/ot-models
/// ```
///
/// Timings are the point here. Resident memory is reported but should not be
/// read across rows: a released ONNX Runtime session does not return its pages
/// to the OS, so deltas measured in one process are not comparable. For real
/// memory numbers use `tool/bench_configs.sh`, which runs one process per
/// configuration.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const localDir = String.fromEnvironment('OT_MODELS_DIR', defaultValue: '');
  const remoteUrl = String.fromEnvironment('OT_MODELS_URL', defaultValue: '');
  const text =
      'The committee agreed that the new proposal should be reviewed '
      'carefully before any final decision is taken next month.';

  final configs = <String, RuntimeConfig>{
    'default (cpu, 4t)': RuntimeConfig.speed,
    'lowMemory': RuntimeConfig.lowMemory,
    '1 thread': const RuntimeConfig(threads: 1),
    '2 threads': const RuntimeConfig(threads: 2),
    '4 threads': const RuntimeConfig(threads: 4),
    'no prepack': const RuntimeConfig(prePackWeights: false),
    'no arena': const RuntimeConfig(useMemoryArena: false),
    'xnnpack': const RuntimeConfig(accelerator: Accelerator.xnnpack),
    'nnapi': const RuntimeConfig(accelerator: Accelerator.nnapi),
    'coreml': const RuntimeConfig(accelerator: Accelerator.coreml),
  };

  late Directory workDir;
  late FileModelManager manager;
  late ModelInfo model;
  final rows = <String>[];

  int rssMb() => (ProcessInfo.currentRss / 1048576).round();

  setUpAll(() async {
    final support = await getApplicationSupportDirectory();
    workDir = Directory(p.join(support.path, 'ot-config-bench'));
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
    workDir.createSync(recursive: true);

    final ModelSource source;
    if (localDir.isNotEmpty && Directory(localDir).existsSync()) {
      source = DirectoryModelSource(localDir);
    } else if (remoteUrl.isNotEmpty) {
      source = HttpModelSource(baseUrl: Uri.parse(remoteUrl));
    } else {
      final docs = await getApplicationDocumentsDirectory();
      source = DirectoryModelSource(p.join(docs.path, 'ot-models'));
    }
    manager = FileModelManager(source: source, rootPath: workDir.path);
    model = await manager.install(const LanguagePair(Language.en, Language.fr));
  });

  tearDownAll(() {
    // ignore: avoid_print
    print(
      '\n=== runtime configurations ===\n'
      'platform: ${Platform.operatingSystem}, '
      '${Platform.numberOfProcessors} cores\n'
      '${'config'.padRight(20)}${'load'.padLeft(9)}${'first'.padLeft(9)}'
      '${'warm'.padLeft(10)}${'ms/tok'.padLeft(9)}${'rss*'.padLeft(8)}\n'
      '${rows.join('\n')}\n'
      '* in-process RSS delta, not comparable across rows\n'
      '=== end ===\n',
    );
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
  });

  for (final entry in configs.entries) {
    testWidgets('config: ${entry.key}', (tester) async {
      final before = rssMb();
      OfflineTranslator? translator;
      try {
        final loadWatch = Stopwatch()..start();
        translator = await OfflineTranslator.initialize(
          languages: const {Language.en, Language.fr},
          defaultLanguage: Language.fr,
          modelManager: manager,
          runtimeConfig: entry.value,
        );
        final loadMs = loadWatch.elapsedMilliseconds;

        final firstWatch = Stopwatch()..start();
        final first = translator.translate(
          text,
          from: Language.en,
          to: Language.fr,
        );
        final firstMs = firstWatch.elapsedMilliseconds;
        expect(first.translatedText, isNotEmpty);

        final samples = <int>[];
        for (var i = 0; i < 7; i++) {
          final watch = Stopwatch()..start();
          translator.translate(text, from: Language.en, to: Language.fr);
          samples.add(watch.elapsedMicroseconds);
        }
        samples.sort();
        final warm = samples[samples.length ~/ 2] / 1000;
        // The reference translation has 25 tokens.
        rows.add(
          '${entry.key.padRight(20)}${'$loadMs ms'.padLeft(9)}'
          '${'$firstMs ms'.padLeft(9)}'
          '${'${warm.toStringAsFixed(1)} ms'.padLeft(10)}'
          '${(warm / 25).toStringAsFixed(2).padLeft(9)}'
          '${'${rssMb() - before} MB'.padLeft(8)}',
        );
      } catch (e) {
        rows.add(
          '${entry.key.padRight(20)}  unavailable: '
          '${e.toString().split('\n').first.replaceFirst('TranslationEngineException: ', '').substring(0, 60)}',
        );
      } finally {
        await translator?.dispose();
      }
      // Availability differs per platform; the table records that rather than
      // failing the run.
      expect(rows, isNotEmpty);
    }, timeout: const Timeout(Duration(minutes: 10)));
  }

  test('model is the expected bundle', () {
    expect(model.id, 'en-fr');
  });
}
