// A pure-Dart harness for the ONNX Runtime FFI layer.
//
//   dart run tool/ffi_harness.dart <models-dir>/en-fr <libonnxruntime.dylib>
//
// It exercises exactly the code path the Flutter engine uses, without a
// Flutter build, which makes the edit-run loop seconds instead of minutes.
// `doc/onnx-runtime.md` shows how to fetch a local runtime.
//
// Modes:
//   (default)   translate the fixture sentences and print timings
//   --leak N    run N translations, printing RSS as it goes
//   --long      translate progressively longer documents
//   --matrix    compare configurations in one process (timings only)
//   --tokens F  check the bundle's tokenizer against reference vectors made by
//               tool/make_tokenizer_vectors.py (no inference, no runtime needed)
//   --config K  run one configuration, so its RSS delta is real; a released
//               ONNX Runtime session does not hand its pages back to the OS,
//               so several configurations in one process report nonsense
//               memory deltas. `tool/bench_configs.sh` loops over the keys.
// ignore_for_file: avoid_print

import 'dart:convert';
import 'dart:io';

// Imported piecemeal rather than through the package barrel: the barrel pulls
// in path_provider, and this harness is meant to run under plain `dart run`.
import 'package:offline_translate/src/core/generation_config.dart';
import 'package:offline_translate/src/core/model_info.dart';
import 'package:offline_translate/src/core/runtime_config.dart';
import 'package:offline_translate/src/engine/native/onnx_runtime.dart';
import 'package:offline_translate/src/engine/onnx_marian_engine.dart';
import 'package:offline_translate/src/tokenizer/marian_tokenizer.dart';
import 'package:offline_translate/src/utils/text_segmenter.dart';
import 'package:path/path.dart' as p;

const List<String> fixtures = <String>[
  'Hello, how are you?',
  'The quick brown fox jumps over the lazy dog.',
  "I would like to book a table for two people at eight o'clock tonight.",
  'Artificial intelligence is transforming the way we build mobile applications.',
  'She said that the meeting had been postponed until next Tuesday because of '
      'the weather.',
];

const String benchText =
    'The committee agreed that the new proposal should be reviewed carefully '
    'before any final decision is taken next month.';

/// Configurations compared in `doc/onnx-runtime.md`.
Map<String, RuntimeConfig> get benchConfigs => <String, RuntimeConfig>{
      'speed': RuntimeConfig.speed,
      'lowMemory': RuntimeConfig.lowMemory,
      'noprepack': const RuntimeConfig(prePackWeights: false),
      'opt-none':
          const RuntimeConfig(graphOptimization: GraphOptimization.none),
      'opt-basic':
          const RuntimeConfig(graphOptimization: GraphOptimization.basic),
      'noarena': const RuntimeConfig(useMemoryArena: false),
      'nomempattern': const RuntimeConfig(useMemoryPattern: false),
      'noarena+noprepack':
          const RuntimeConfig(useMemoryArena: false, prePackWeights: false),
      'noarena+optnone': const RuntimeConfig(
          useMemoryArena: false, graphOptimization: GraphOptimization.none),
      'noarena+nopattern':
          const RuntimeConfig(useMemoryArena: false, useMemoryPattern: false),
      'all-off': const RuntimeConfig(
          useMemoryArena: false,
          useMemoryPattern: false,
          prePackWeights: false,
          graphOptimization: GraphOptimization.none),
      't1': const RuntimeConfig(threads: 1),
      't2': const RuntimeConfig(threads: 2),
      't4': const RuntimeConfig(threads: 4),
      't6': const RuntimeConfig(threads: 6),
      't8': const RuntimeConfig(threads: 8),
      'xnnpack': const RuntimeConfig(accelerator: Accelerator.xnnpack),
      'nnapi': const RuntimeConfig(accelerator: Accelerator.nnapi),
      'coreml': const RuntimeConfig(accelerator: Accelerator.coreml),
    };

int rssMb() => (ProcessInfo.currentRss / 1048576).round();

String pad(Object value, int width) => value.toString().padLeft(width);

ModelInfo loadManifest(String dir) =>
    ModelInfo.parse(File(p.join(dir, 'manifest.json')).readAsStringSync(),
        path: dir);

Future<OnnxMarianEngine> open(String dir, RuntimeConfig config) async {
  final engine = OnnxMarianEngine(loadManifest(dir), runtimeConfig: config);
  await engine.load();
  return engine;
}

String translate(OnnxMarianEngine engine, String text,
    [GenerationConfig config = const GenerationConfig()]) {
  final ids = engine.encodeText(text);
  final out = engine.generate(ids, config);
  return engine.decodeTokens(out.tokens);
}

String words(int count) {
  const sample = 'The quick brown fox jumps over the lazy dog near the river '
      'bank while the sun slowly sets behind the distant hills and the evening '
      'air turns cold. ';
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

Future<void> main(List<String> args) async {
  if (args.length == 1 && args.first == '--header') {
    print('${'config'.padRight(14)}${pad('load', 9)}${pad('rss', 9)}'
        '${pad('peak', 8)}${pad('first', 9)}${pad('warm', 9)}'
        '${pad('ms/tok', 8)}');
    return;
  }
  // The tokenizer check needs no runtime at all, so it runs before the library
  // is resolved: a direction can be validated on a machine with no ONNX Runtime.
  if (args.length >= 3 && args[1] == '--tokens') {
    exit(_checkTokens(args[0], args[2]));
  }
  if (args.length < 2) {
    stderr.writeln('usage: dart run tool/ffi_harness.dart <model-dir> '
        '<libonnxruntime> [--leak N | --long | --matrix | --config KEY]\n'
        '   or: dart run tool/ffi_harness.dart <model-dir> --tokens <file>');
    exit(64);
  }
  final dir = args[0];
  OrtLibrary.overrideLibraryPath = args[1];
  final mode = args.length > 2 ? args[2] : '';

  if (mode != '--config') {
    print('ONNX Runtime ${OrtLibrary.instance.versionString} '
        '(API v${OrtLibrary.instance.version})');
    print('platform ${Platform.operatingSystem}, '
        '${Platform.numberOfProcessors} cores, RSS ${rssMb()} MB\n');
  }

  switch (mode) {
    case '--leak':
      await _leak(dir, int.parse(args[3]));
    case '--long':
      await _long(dir);
    case '--matrix':
      await _matrix(dir);
    case '--config':
      await _single(dir, args[3], args.length > 4 ? int.parse(args[4]) : 9);
    default:
      await _smoke(dir);
  }
}

/// Compares the Dart tokenizer against reference vectors, returning an exit
/// code so a build script can gate on it.
int _checkTokens(String dir, String vectorsPath) {
  final tokenizer = MarianTokenizer.fromAssets(
    spmBytes: File(p.join(dir, 'source.spm')).readAsBytesSync(),
    vocabJson: File(p.join(dir, 'vocab.json')).readAsStringSync(),
  );
  final data =
      jsonDecode(File(vectorsPath).readAsStringSync()) as Map<String, dynamic>;
  final pair = data['pair'] as String;
  final vectors = data['vectors'] as List<dynamic>;

  if (tokenizer.vocabSize != data['vocab_size']) {
    stderr.writeln('$pair: vocab size ${tokenizer.vocabSize} != '
        '${data['vocab_size']} from the manifest');
    return 1;
  }

  var pieceFailures = 0;
  var idFailures = 0;
  String? firstFailure;
  for (final entry in vectors) {
    final vector = entry as Map<String, dynamic>;
    final text = vector['text'] as String;
    final expectedPieces = (vector['pieces'] as List<dynamic>).cast<String>();
    final expectedIds = (vector['ids'] as List<dynamic>).cast<int>();
    final pieces = tokenizer.tokenize(text);
    final ids = tokenizer.encode(text);
    if (pieces.join('\u0000') != expectedPieces.join('\u0000')) {
      pieceFailures++;
      firstFailure ??= jsonEncode(text);
    }
    if (ids.join(',') != expectedIds.join(',')) {
      idFailures++;
      firstFailure ??= jsonEncode(text);
    }
  }

  if (pieceFailures == 0 && idFailures == 0) {
    stdout.writeln('$pair tokenizer: ${vectors.length}/${vectors.length} '
        'vectors match (vocab ${tokenizer.vocabSize}, eos ${tokenizer.eosId}, '
        'unk ${tokenizer.unknownId}, pad ${tokenizer.padId})');
    return 0;
  }
  stderr.writeln('$pair tokenizer: $pieceFailures piece and $idFailures id '
      'mismatches out of ${vectors.length}; first divergence: $firstFailure');
  return 1;
}

Future<void> _smoke(String dir) async {
  final before = rssMb();
  final loadWatch = Stopwatch()..start();
  final engine = await open(dir, RuntimeConfig.speed);
  loadWatch.stop();
  print('load ${loadWatch.elapsedMilliseconds} ms, '
      'RSS ${rssMb()} MB (+${rssMb() - before})');

  for (final text in fixtures) {
    final watch = Stopwatch()..start();
    final ids = engine.encodeText(text);
    final out = engine.generate(ids, const GenerationConfig());
    watch.stop();
    final perToken = watch.elapsedMicroseconds / 1000 / out.tokens.length;
    print('\nEN: $text\nFR: ${engine.decodeTokens(out.tokens)}');
    print('    ${out.tokens.length} tokens, ${watch.elapsedMilliseconds} ms '
        '(${perToken.toStringAsFixed(1)} ms/tok), '
        'encoder ${(out.encodeMicros / 1000).toStringAsFixed(1)} ms');
  }
  print('\nRSS after translations ${rssMb()} MB');
  await engine.dispose();
  print('RSS after dispose ${rssMb()} MB');
}

Future<void> _leak(String dir, int iterations) async {
  final engine = await open(dir, RuntimeConfig.speed);
  // Warm up so the first-run arena growth is not counted as a leak.
  for (var i = 0; i < 5; i++) {
    translate(engine, benchText);
  }
  final baseline = rssMb();
  var tokens = 0;
  print('baseline after warm-up: $baseline MB');
  final watch = Stopwatch()..start();
  for (var i = 1; i <= iterations; i++) {
    final ids = engine.encodeText(benchText);
    tokens += engine.generate(ids, const GenerationConfig()).tokens.length;
    if (i == 10 ||
        i == 50 ||
        i == 100 ||
        i == 500 ||
        i == 1000 ||
        i == iterations) {
      final average = watch.elapsedMicroseconds / 1000 / i;
      print('after ${pad(i, 5)} translations (${pad(tokens, 6)} tokens): '
          '${pad(rssMb(), 4)} MB  delta ${pad(rssMb() - baseline, 5)} MB  '
          '${average.toStringAsFixed(1)} ms avg');
    }
  }
  await engine.dispose();
  print('after dispose: ${rssMb()} MB');
}

Future<void> _long(String dir) async {
  final engine = await open(dir, RuntimeConfig.speed);
  const segmenter = TextSegmenter();
  const config = GenerationConfig();
  print('${pad('words', 7)}${pad('chars', 9)}${pad('chunks', 8)}'
      '${pad('tokens', 8)}${pad('before', 8)}${pad('peak', 7)}'
      '${pad('after', 7)}${pad('seconds', 9)}${pad('tok/s', 8)}');
  for (final count in <int>[100, 500, 1000, 5000, 10000, 20000]) {
    final text = paragraphs(count);
    final before = rssMb();
    var peak = before;
    var tokens = 0;
    final watch = Stopwatch()..start();
    final chunks = segmenter.split(text,
        maxTokens: config.maxInputTokens,
        countTokens: (s) => engine.encodeText(s).length);
    for (final chunk in chunks) {
      final ids = engine.encodeText(chunk.text);
      tokens += engine.generate(ids, config).tokens.length;
      final now = rssMb();
      if (now > peak) peak = now;
    }
    watch.stop();
    final seconds = watch.elapsedMilliseconds / 1000;
    print('${pad(count, 7)}${pad(text.length, 9)}${pad(chunks.length, 8)}'
        '${pad(tokens, 8)}${pad(before, 8)}${pad(peak, 7)}${pad(rssMb(), 7)}'
        '${pad(seconds.toStringAsFixed(1), 9)}'
        '${pad((tokens / seconds).round(), 8)}');
  }
  await engine.dispose();
}

/// Runs one configuration in this process, so the RSS delta is real.
Future<void> _single(String dir, String key, int samples) async {
  final config = benchConfigs[key];
  if (config == null) {
    stderr.writeln('unknown config "$key"; known: '
        '${benchConfigs.keys.join(", ")}');
    exit(64);
  }
  final before = rssMb();
  try {
    final loadWatch = Stopwatch()..start();
    final engine = await open(dir, config);
    final loadMs = loadWatch.elapsedMilliseconds;

    final firstWatch = Stopwatch()..start();
    translate(engine, benchText);
    final firstMs = firstWatch.elapsedMilliseconds;
    final afterFirst = rssMb() - before;

    final times = <int>[];
    var tokens = 1;
    var peak = rssMb();
    for (var i = 0; i < samples; i++) {
      final watch = Stopwatch()..start();
      final ids = engine.encodeText(benchText);
      tokens = engine.generate(ids, const GenerationConfig()).tokens.length;
      times.add(watch.elapsedMicroseconds);
      final now = rssMb();
      if (now > peak) peak = now;
    }
    times.sort();
    final warm = times[times.length ~/ 2] / 1000;
    print('${key.padRight(14)}${pad('$loadMs ms', 9)}'
        '${pad('$afterFirst MB', 9)}${pad('${peak - before} MB', 8)}'
        '${pad('$firstMs ms', 9)}${pad('${warm.toStringAsFixed(1)} ms', 9)}'
        '${pad((warm / tokens).toStringAsFixed(2), 8)}');
    await engine.dispose();
  } catch (e) {
    print('${key.padRight(14)}  unavailable: ${_short(e)}');
  }
}

Future<void> _matrix(String dir) async {
  print('${'config'.padRight(14)}${pad('load', 9)}${pad('first', 9)}'
      '${pad('warm', 9)}${pad('ms/tok', 8)}   (memory needs --config)');
  for (final entry in benchConfigs.entries) {
    try {
      final loadWatch = Stopwatch()..start();
      final engine = await open(dir, entry.value);
      final loadMs = loadWatch.elapsedMilliseconds;

      final firstWatch = Stopwatch()..start();
      translate(engine, benchText);
      final firstMs = firstWatch.elapsedMilliseconds;

      final times = <int>[];
      var tokens = 1;
      for (var i = 0; i < 9; i++) {
        final watch = Stopwatch()..start();
        final ids = engine.encodeText(benchText);
        tokens = engine.generate(ids, const GenerationConfig()).tokens.length;
        times.add(watch.elapsedMicroseconds);
      }
      times.sort();
      final warm = times[times.length ~/ 2] / 1000;
      print('${entry.key.padRight(14)}${pad('$loadMs ms', 9)}'
          '${pad('$firstMs ms', 9)}${pad('${warm.toStringAsFixed(1)} ms', 9)}'
          '${pad((warm / tokens).toStringAsFixed(2), 8)}');
      await engine.dispose();
    } catch (e) {
      print('${entry.key.padRight(14)}  unavailable: ${_short(e)}');
    }
  }
}

String _short(Object error) => error
    .toString()
    .split('\n')
    .first
    .replaceFirst('TranslationEngineException: ', '');
