import 'dart:async';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:offline_translate/offline_translate.dart';

import 'model_setup.dart';

/// Headless self-check used by `tool/offline_proof.sh`.
///
/// Enabled with `--dart-define=OT_AUTORUN=1`. On launch the app installs the
/// model if it is missing, translates a short and a long text, and prints one
/// machine-readable line per step so the shell script can assert on them
/// without any test harness — which matters because the harness reinstalls the
/// app, and the whole point of this check is that an *already installed* model
/// keeps working after a relaunch with the network switched off.
class AutorunReport {
  /// Creates a report.
  const AutorunReport(this.lines);

  /// The `OT_AUTORUN` lines that were printed.
  final List<String> lines;
}

/// Whether autorun mode was requested.
///
/// Accepts any non-empty value, so both `--dart-define=OT_AUTORUN=1` and
/// `--dart-define=OT_AUTORUN=true` work.
const bool autorunEnabled =
    String.fromEnvironment('OT_AUTORUN', defaultValue: '') != '';

/// `--dart-define=OT_AUTORUN=bench` adds the benchmark to the self-check.
///
/// This is the only way to get **profile or release mode** numbers on a real
/// device: `flutter test integration_test` always builds in debug, which
/// inflates every Dart-side cost. Driving the app itself instead
/// (`flutter run --profile --dart-define=OT_AUTORUN=bench`) measures what a
/// shipped application would actually do.
const bool autorunBenchmark =
    String.fromEnvironment('OT_AUTORUN', defaultValue: '') == 'bench';

/// Direction to exercise, e.g. `--dart-define=OT_AUTORUN_PAIR=de-es`.
const String autorunPair =
    String.fromEnvironment('OT_AUTORUN_PAIR', defaultValue: 'en-fr');

/// Runs the self-check and prints its findings.
Future<AutorunReport> runSelfCheck() async {
  final lines = <String>[];
  void emit(String line) {
    lines.add(line);
    debugPrint('OT_AUTORUN $line');
  }

  OfflineTranslator? translator;
  try {
    final resolved = await DemoModelSources.resolve();
    emit('source=${resolved.description}');

    translator = await OfflineTranslator.initialize(
      modelSource: resolved.source,
    );

    // Is the network actually reachable from this process right now?
    var online = false;
    try {
      final socket = await Socket.connect(
        'example.com',
        80,
        timeout: const Duration(seconds: 4),
      );
      socket.destroy();
      online = true;
    } catch (_) {
      online = false;
    }
    emit('online=$online');

    final pair = LanguagePair(
      Language.parse(autorunPair.split('-').first),
      Language.parse(autorunPair.split('-').last),
    );
    final already = await translator.isModelAvailable(
      from: pair.from,
      to: pair.to,
    );
    emit('installed_before=$already');

    if (!already) {
      final info = await translator.installModel(from: pair.from, to: pair.to);
      emit('installed=${info.id} size=${info.size} license=${info.license}');
    }

    await translator.preload(from: pair.from, to: pair.to);
    emit('loaded=${pair.id}');

    final short = translator.translateSync(
      text: 'Hello, how are you?',
      from: pair.from,
      to: pair.to,
    );
    emit(
      'sync_ms=${short.duration.inMilliseconds} '
      'text=${short.translatedText}',
    );

    final long = await translator.translate(
      text:
          'The network is switched off. This sentence is translated '
          'entirely on the device.\n\n'
          'Nothing leaves the phone, and no server is involved at any point.',
      from: pair.from,
      to: pair.to,
    );
    emit(
      'async_ms=${long.duration.inMilliseconds} '
      'chunks=${long.chunkCount} '
      'text=${long.translatedText.replaceAll('\n', ' | ')}',
    );

    if (autorunBenchmark) {
      await _benchmark(translator, pair, emit);
    }

    emit('status=OK');
  } catch (e) {
    emit('status=FAILED error=$e');
  } finally {
    await translator?.dispose();
  }
  return AutorunReport(lines);
}

/// Timings and memory for the current build mode, printed as `OT_AUTORUN`
/// lines so `tool/device_bench.sh` can collect them.
Future<void> _benchmark(
  OfflineTranslator translator,
  LanguagePair pair,
  void Function(String) emit,
) async {
  String words(int count) {
    const sample = 'The quick brown fox jumps over the lazy dog near the river '
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

  int rssMb() => (ProcessInfo.currentRss / 1048576).round();

  emit('bench_mode=${_buildMode()} cores=${Platform.numberOfProcessors} '
      'rss=${rssMb()}MB');

  const short = 'Hello world';
  const sentence = 'The committee agreed that the new proposal should be '
      'reviewed carefully before any final decision is taken next month.';

  for (final entry in <(String, String)>[
    ('hello', short),
    ('sentence20', sentence),
    ('words100', words(100)),
    ('words500', words(500)),
  ]) {
    final samples = <int>[];
    var chunks = 1;
    for (var i = 0; i < 5; i++) {
      final watch = Stopwatch()..start();
      final result = translator.translateSync(
          text: entry.$2, from: pair.from, to: pair.to);
      samples.add(watch.elapsedMicroseconds);
      chunks = result.chunkCount;
    }
    samples.sort();
    emit('bench_${entry.$1}='
        '${(samples[2] / 1000).toStringAsFixed(1)}ms chunks=$chunks');
  }

  // Repeated translation, to show memory does not creep on this device.
  final baseline = rssMb();
  for (var i = 0; i < 200; i++) {
    translator.translateSync(text: sentence, from: pair.from, to: pair.to);
  }
  emit('bench_leak_200x=${rssMb() - baseline}MB rss=${rssMb()}MB');

  // A long document through the async API, timing the caller's own stalls.
  final document = List<String>.generate(30, (_) => words(60)).join('\n\n');
  final gaps = <int>[];
  var last = DateTime.now();
  final ticker = Timer.periodic(const Duration(milliseconds: 8), (_) {
    final now = DateTime.now();
    gaps.add(now.difference(last).inMilliseconds);
    last = now;
  });
  final watch = Stopwatch()..start();
  final result =
      await translator.translate(text: document, from: pair.from, to: pair.to);
  watch.stop();
  ticker.cancel();
  gaps.sort();
  emit('bench_async_doc=${watch.elapsedMilliseconds}ms '
      'chars=${document.length} chunks=${result.chunkCount} '
      'ticks=${gaps.length} worst_stall=${gaps.isEmpty ? -1 : gaps.last}ms '
      'rss=${rssMb()}MB');
}

String _buildMode() {
  if (const bool.fromEnvironment('dart.vm.product')) return 'release';
  var profile = true;
  assert(() {
    profile = false;
    return true;
  }());
  return profile ? 'profile' : 'debug';
}
