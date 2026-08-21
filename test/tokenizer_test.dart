import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translate/src/tokenizer/marian_tokenizer.dart';

void main() {
  late MarianTokenizer tokenizer;
  late List<dynamic> vectors;

  setUpAll(() {
    tokenizer = MarianTokenizer.fromAssets(
      spmBytes: File('test/fixtures/source.spm').readAsBytesSync(),
      vocabJson: File('test/fixtures/vocab.json').readAsStringSync(),
    );
    vectors = jsonDecode(
      File('test/fixtures/tokenizer_vectors.json').readAsStringSync(),
    ) as List<dynamic>;
  });

  test('vocabulary metadata matches the Marian checkpoint', () {
    expect(tokenizer.vocabSize, 59514);
    expect(tokenizer.eosId, 0);
    expect(tokenizer.unknownId, 1);
    expect(tokenizer.padId, 59513);
  });

  test('segmentation matches the SentencePiece reference', () {
    for (final v in vectors) {
      final c = v as Map<String, dynamic>;
      final text = c['text'] as String;
      final expected = (c['pieces'] as List<dynamic>)
          .map((dynamic e) => e as String)
          .toList();
      expect(tokenizer.tokenize(text), expected,
          reason: 'pieces mismatch for ${jsonEncode(text)}');
    }
  });

  test('encoding matches transformers.MarianTokenizer', () {
    for (final v in vectors) {
      final c = v as Map<String, dynamic>;
      final text = c['text'] as String;
      final expected =
          (c['ids'] as List<dynamic>).map((dynamic e) => e as int).toList();
      expect(tokenizer.encode(text).toList(), expected,
          reason: 'ids mismatch for ${jsonEncode(text)}');
    }
  });

  test('matches SentencePiece on a 1793-case fuzz corpus', () {
    final fuzz = jsonDecode(
      File('test/fixtures/tokenizer_fuzz.json').readAsStringSync(),
    ) as List<dynamic>;
    var failures = 0;
    String? firstFailure;
    for (final v in fuzz) {
      final c = v as Map<String, dynamic>;
      final text = c['text'] as String;
      final expectedIds =
          (c['ids'] as List<dynamic>).map((dynamic e) => e as int).toList();
      final expectedPieces = (c['pieces'] as List<dynamic>)
          .map((dynamic e) => e as String)
          .toList();
      if (tokenizer.tokenize(text).join('\u0000') !=
              expectedPieces.join('\u0000') ||
          tokenizer.encode(text).join(',') != expectedIds.join(',')) {
        failures++;
        firstFailure ??= jsonEncode(text);
      }
    }
    expect(failures, 0, reason: 'first divergence: $firstFailure');
  });

  test('round-trips text through encode/decode', () {
    const text = 'Hello, how are you?';
    final ids = tokenizer.encode(text);
    expect(tokenizer.decode(ids), text);
  });

  test('decode drops special tokens by default', () {
    final ids = <int>[tokenizer.padId, ...tokenizer.encode('Hello')];
    expect(tokenizer.decode(ids), 'Hello');
  });
}
