import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Exercises every direction that is present in the model source.
///
/// ```sh
/// flutter test integration_test/all_directions_test.dart -d macos \
///   --dart-define=OT_MODELS_DIR=$HOME/ot-models
/// ```
///
/// Directions that are not published yet are skipped rather than failed, so
/// this suite is useful while the catalogue is still being filled in.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const localDir = String.fromEnvironment('OT_MODELS_DIR', defaultValue: '');
  const remoteUrl = String.fromEnvironment('OT_MODELS_URL', defaultValue: '');

  /// One sentence per source language, and a word that must appear in a
  /// correct translation of it. The marker is deliberately a content word that
  /// every reasonable rendering keeps, not an exact string match — greedy
  /// decoding is allowed to phrase things its own way.
  const samples =
      <
        Language,
        ({String text, List<Language> to, Map<Language, String> marker})
      >{
        Language.en: (
          text: 'The meeting has been postponed until next Tuesday.',
          to: <Language>[Language.fr, Language.es, Language.de],
          marker: <Language, String>{
            Language.fr: 'mardi',
            Language.es: 'martes',
            Language.de: 'Dienstag',
          },
        ),
        Language.fr: (
          text: 'La réunion a été reportée à mardi prochain.',
          to: <Language>[Language.en, Language.es, Language.de],
          marker: <Language, String>{
            Language.en: 'Tuesday',
            Language.es: 'martes',
            Language.de: 'Dienstag',
          },
        ),
        Language.es: (
          text: 'La reunión se ha aplazado hasta el próximo martes.',
          to: <Language>[Language.en, Language.fr, Language.de],
          marker: <Language, String>{
            Language.en: 'Tuesday',
            Language.fr: 'mardi',
            Language.de: 'Dienstag',
          },
        ),
        Language.de: (
          text: 'Die Sitzung wurde auf nächsten Dienstag verschoben.',
          to: <Language>[Language.en, Language.fr, Language.es],
          marker: <Language, String>{
            Language.en: 'Tuesday',
            Language.fr: 'mardi',
            Language.es: 'martes',
          },
        ),
      };

  late Directory workDir;
  late FileModelManager manager;
  late OfflineTranslator translator;
  final report = <String>[];
  final available = <LanguagePair>[];

  setUpAll(() async {
    final support = await getApplicationSupportDirectory();
    workDir = Directory(p.join(support.path, 'ot-all-directions'));
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
    translator = await OfflineTranslator.initialize(
      modelManager: manager,
      // Twelve resident models would be ~2 GB. One at a time is the point.
      maxLoadedModels: 1,
    );
  });

  tearDownAll(() async {
    await translator.dispose();
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
    // ignore: avoid_print
    print(
      '\n=== directions ===\n${report.join('\n')}\n'
      '${available.length}/12 available\n=== end ===\n',
    );
  });

  for (final entry in samples.entries) {
    final from = entry.key;
    for (final to in entry.value.to) {
      final pair = LanguagePair(from, to);
      testWidgets(pair.id, (tester) async {
        final ModelInfo info;
        try {
          info = await translator.installModel(from: from, to: to);
        } on ModelDownloadException {
          report.add('${pair.id.padRight(6)} not published yet — skipped');
          return;
        }
        available.add(pair);

        expect(info.id, pair.id);
        expect(info.baseModel, 'Helsinki-NLP/opus-mt-${pair.id}');
        expect(info.license, anyOf('Apache-2.0', 'CC-BY-4.0'));
        expect(info.architecture.decoderLayers, greaterThan(0));
        expect(info.architecture.vocabSize, greaterThan(1000));
        // The whole bundle is re-hashed, not just size-checked.
        await manager.verify(pair);

        final watch = Stopwatch()..start();
        final result = await translator.translateLong(
          entry.value.text,
          from: from,
          to: to,
        );
        watch.stop();

        expect(result.translatedText.trim(), isNotEmpty);
        expect(result.sourceLanguage, from);
        expect(result.targetLanguage, to);
        expect(result.truncated, isFalse);

        final marker = entry.value.marker[to]!;
        final matched = result.translatedText.toLowerCase().contains(
          marker.toLowerCase(),
        );
        report.add(
          '${pair.id.padRight(6)} '
          '${(info.size / 1048576).toStringAsFixed(0).padLeft(4)} MB  '
          '${info.license.padRight(11)} '
          '${watch.elapsedMilliseconds.toString().padLeft(5)} ms  '
          '${matched ? '✓' : '?'} ${result.translatedText}',
        );
        expect(
          matched,
          isTrue,
          reason:
              '${pair.id}: expected "$marker" in '
              '"${result.translatedText}"',
        );

        // Only one model may stay resident.
        expect(translator.loadedModels.length, lessThanOrEqualTo(1));
      }, timeout: const Timeout(Duration(minutes: 10)));
    }
  }

  testWidgets(
    'installed models list what was installed',
    (tester) async {
      final installed = await translator.installedModels();
      expect(
        installed.map((m) => m.id).toSet(),
        available.map((p) => p.id).toSet(),
      );
      for (final model in installed) {
        expect(model.quantization, 'int8');
        expect(model.version, isNotEmpty);
        expect(model.checksum, hasLength(64));
      }
    },
    timeout: const Timeout(Duration(minutes: 5)),
  );
}
