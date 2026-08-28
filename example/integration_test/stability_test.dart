import 'dart:async';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Long-run behaviour: does memory stay flat, and does the UI isolate stay free?
///
/// ```sh
/// flutter test integration_test/stability_test.dart -d macos \
///   --dart-define=OT_MODELS_DIR=$HOME/ot-models
/// ```
///
/// Add `--dart-define=OT_STABILITY_ITERATIONS=1000` for the full run; the
/// default is smaller so the suite stays usable in CI.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const localDir = String.fromEnvironment('OT_MODELS_DIR', defaultValue: '');
  const remoteUrl = String.fromEnvironment('OT_MODELS_URL', defaultValue: '');
  const shortIterations = int.fromEnvironment(
    'OT_STABILITY_ITERATIONS',
    defaultValue: 200,
  );
  const paragraphIterations = int.fromEnvironment(
    'OT_STABILITY_PARAGRAPHS',
    defaultValue: 20,
  );

  const sentence =
      'The committee agreed that the new proposal should be '
      'reviewed carefully before any final decision is taken next month.';

  String words(int count) {
    const sample =
        'The quick brown fox jumps over the lazy dog near the river '
        'bank while the sun slowly sets behind the distant hills and the '
        'evening air turns cold. ';
    final buffer = StringBuffer();
    var produced = 0;
    while (produced < count) {
      buffer.write(sample);
      produced += 28;
    }
    return buffer.toString().trim();
  }

  String paragraphs(int wordCount) {
    final buffer = StringBuffer();
    var produced = 0;
    while (produced < wordCount) {
      buffer
        ..write(words(60))
        ..write('\n\n');
      produced += 60;
    }
    return buffer.toString().trim();
  }

  int rssMb() => (ProcessInfo.currentRss / 1048576).round();

  late Directory workDir;
  late OfflineTranslator translator;
  final report = <String>[];

  setUpAll(() async {
    final support = await getApplicationSupportDirectory();
    workDir = Directory(p.join(support.path, 'ot-stability'));
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
    workDir.createSync(recursive: true);

    final ModelSource source;
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
    final manager = FileModelManager(source: source, rootPath: workDir.path);
    await manager.install(const LanguagePair(Language.en, Language.fr));
    translator = await OfflineTranslator.initialize(
      languages: const {Language.en, Language.fr},
      defaultLanguage: Language.fr,
      modelManager: manager,
    );
  });

  tearDownAll(() async {
    await translator.dispose();
    if (workDir.existsSync()) workDir.deleteSync(recursive: true);
    // ignore: avoid_print
    print('\n=== stability ===\n${report.join('\n')}\n=== end ===\n');
  });

  testWidgets(
    'memory stays flat over $shortIterations short translations',
    (tester) async {
      // Warm up so one-off allocator growth is not read as a leak.
      for (var i = 0; i < 5; i++) {
        translator.translate(sentence);
      }
      final baseline = rssMb();
      report.add('short translations: baseline $baseline MB');

      var tokens = 0;
      var peak = baseline;
      final watch = Stopwatch()..start();
      for (var i = 1; i <= shortIterations; i++) {
        final result = translator.translate(sentence);
        tokens += result.translatedText.length;
        final now = rssMb();
        if (now > peak) peak = now;
        if (i == 10 || i == 100 || i == shortIterations) {
          report.add(
            '  after $i: ${rssMb()} MB '
            '(delta ${rssMb() - baseline} MB, '
            '${(watch.elapsedMilliseconds / i).toStringAsFixed(1)} ms avg)',
          );
        }
      }
      final growth = rssMb() - baseline;
      report.add(
        '  peak $peak MB, final ${rssMb()} MB, growth $growth MB, '
        '$tokens chars produced',
      );

      // A per-token native leak of the size the old binding had would show up as
      // tens of megabytes here. Allow generous headroom for allocator behaviour
      // and the Dart heap, but not unbounded growth.
      expect(
        growth,
        lessThan(80),
        reason:
            'resident memory grew $growth MB over $shortIterations '
            'translations, which suggests a leak',
      );
    },
    timeout: const Timeout(Duration(minutes: 30)),
  );

  testWidgets('memory stays flat over $paragraphIterations paragraph '
      'translations', (tester) async {
    final document = paragraphs(300);
    await translator.translateLong(document);
    final baseline = rssMb();
    report.add(
      'paragraph translations: baseline $baseline MB, '
      '${document.length} chars each',
    );

    var peak = baseline;
    final watch = Stopwatch()..start();
    for (var i = 1; i <= paragraphIterations; i++) {
      await translator.translateLong(document);
      final now = rssMb();
      if (now > peak) peak = now;
      if (i == 5 || i == paragraphIterations) {
        report.add(
          '  after $i: ${rssMb()} MB '
          '(delta ${rssMb() - baseline} MB, '
          '${(watch.elapsedMilliseconds / i / 1000).toStringAsFixed(1)} s '
          'avg)',
        );
      }
    }
    final growth = rssMb() - baseline;
    report.add('  peak $peak MB, final ${rssMb()} MB, growth $growth MB');
    expect(
      growth,
      lessThan(120),
      reason:
          'resident memory grew $growth MB over $paragraphIterations '
          'document translations',
    );
  }, timeout: const Timeout(Duration(minutes: 60)));

  testWidgets(
    'translate() keeps the calling isolate responsive',
    (tester) async {
      final document = paragraphs(2000);

      // A ticker on the calling isolate, sampling how long each gap is. If
      // inference ran here, gaps would be hundreds of milliseconds wide.
      final gaps = <int>[];
      var last = DateTime.now();
      final ticker = Timer.periodic(const Duration(milliseconds: 8), (_) {
        final now = DateTime.now();
        gaps.add(now.difference(last).inMilliseconds);
        last = now;
      });

      final watch = Stopwatch()..start();
      final result = await translator.translateLong(document);
      watch.stop();
      ticker.cancel();

      expect(result.translatedText.trim(), isNotEmpty);
      gaps.sort();
      final worst = gaps.isEmpty ? -1 : gaps.last;
      final median = gaps.isEmpty ? -1 : gaps[gaps.length ~/ 2];
      report.add(
        'responsiveness: ${document.length} chars in '
        '${(watch.elapsedMilliseconds / 1000).toStringAsFixed(1)} s, '
        '${gaps.length} ticks, median gap $median ms, worst gap $worst ms',
      );

      expect(
        gaps.length,
        greaterThan(20),
        reason: 'the calling isolate barely ran during translate()',
      );
      // With inference on the calling isolate this gap is the cost of a whole
      // chunk — over a second on a phone, and 1 073 ms measured on a laptop for
      // 100 words. Off-isolate it is a scheduling hiccup: 40 ms on macOS, 116 ms
      // on a contended 4-core emulator. The threshold sits between the two.
      expect(
        worst,
        lessThan(400),
        reason:
            'the calling isolate stalled for $worst ms during translate(), '
            'so inference is probably not running off-isolate',
      );
    },
    timeout: const Timeout(Duration(minutes: 30)),
  );

  testWidgets(
    'translateSync is the one that blocks, by design',
    (tester) async {
      final sentenceGaps = <int>[];
      var last = DateTime.now();
      final ticker = Timer.periodic(const Duration(milliseconds: 8), (_) {
        final now = DateTime.now();
        sentenceGaps.add(now.difference(last).inMilliseconds);
        last = now;
      });
      // A deliberate paragraph through the synchronous API: this is the case the
      // documentation tells callers to avoid, and the numbers here are what that
      // warning is based on.
      final watch = Stopwatch()..start();
      translator.translate(words(100));
      watch.stop();
      await Future<void>.delayed(const Duration(milliseconds: 50));
      ticker.cancel();

      sentenceGaps.sort();
      report.add(
        'translateSync on 100 words: ${watch.elapsedMilliseconds} ms, '
        'worst tick gap ${sentenceGaps.isEmpty ? -1 : sentenceGaps.last} ms',
      );
      expect(watch.elapsedMilliseconds, greaterThan(0));
    },
    timeout: const Timeout(Duration(minutes: 10)),
  );
}
