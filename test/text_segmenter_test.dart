import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translate/offline_translate.dart';

/// Stand-in for the real tokenizer: one token per whitespace-separated word,
/// plus one for the end-of-sequence marker the encoder appends.
int wordTokens(String s) =>
    s.trim().isEmpty ? 1 : s.trim().split(RegExp(r'\s+')).length + 1;

void main() {
  const segmenter = TextSegmenter();

  List<TextChunk> split(String text, int maxTokens) =>
      segmenter.split(text, maxTokens: maxTokens, countTokens: wordTokens);

  test('returns nothing for empty or blank input', () {
    expect(split('', 10), isEmpty);
    expect(split('   \n\n  ', 10), isEmpty);
  });

  test('keeps short text as a single chunk', () {
    final chunks = split('Hello world', 10);
    expect(chunks, hasLength(1));
    expect(chunks.single.text, 'Hello world');
    expect(chunks.single.separator, '');
  });

  test('splits on paragraphs first', () {
    final chunks = split('First paragraph.\n\nSecond paragraph.', 10);
    expect(chunks.map((c) => c.text).toList(),
        <String>['First paragraph.', 'Second paragraph.']);
    expect(chunks.first.separator, '\n\n');
  });

  test('reassembly reproduces the original layout', () {
    const text = 'One.\n\nTwo.\nThree.\n\nFour.';
    final chunks = split(text, 10);
    expect(
        segmenter.reassemble(chunks, chunks.map((c) => c.text).toList()), text);
  });

  test('splits an oversized paragraph on sentence boundaries', () {
    const text = 'Alpha beta gamma delta. Epsilon zeta eta theta. '
        'Iota kappa lambda mu.';
    final chunks = split(text, 6);
    expect(chunks.length, greaterThanOrEqualTo(3));
    for (final chunk in chunks) {
      expect(wordTokens(chunk.text), lessThanOrEqualTo(6),
          reason: 'chunk over budget: ${chunk.text}');
    }
    expect(
        segmenter.reassemble(chunks, chunks.map((c) => c.text).toList()), text);
  });

  test('packs several short sentences into one chunk', () {
    const text = 'One. Two. Three. Four. Five. Six.';
    final chunks = split(text, 8);
    expect(chunks.length, lessThan(6));
    for (final chunk in chunks) {
      expect(wordTokens(chunk.text), lessThanOrEqualTo(8));
    }
  });

  test('falls back to clauses when a sentence is too long', () {
    const text = 'Alpha beta, gamma delta, epsilon zeta, eta theta.';
    final chunks = split(text, 4);
    expect(chunks.length, greaterThan(1));
    for (final chunk in chunks) {
      expect(wordTokens(chunk.text), lessThanOrEqualTo(4));
    }
    expect(
        segmenter.reassemble(chunks, chunks.map((c) => c.text).toList()), text);
  });

  test('falls back to words when a clause is still too long', () {
    final text = List<String>.filled(40, 'word').join(' ');
    final chunks = split(text, 5);
    expect(chunks.length, greaterThan(5));
    for (final chunk in chunks) {
      expect(wordTokens(chunk.text), lessThanOrEqualTo(5));
    }
    expect(
        segmenter.reassemble(chunks, chunks.map((c) => c.text).toList()), text);
  });

  test('slices a single enormous token', () {
    final text = 'a' * 1000;
    final chunks = split(text, 1);
    expect(chunks.length, greaterThan(1));
    expect(chunks.map((c) => c.text).join(), text);
  });

  test('handles a long document without losing characters', () {
    final buffer = StringBuffer();
    for (var i = 0; i < 40; i++) {
      buffer.write('Paragraph $i has a first sentence. '
          'It also has a second sentence, with a clause. '
          'And a third one.\n\n');
    }
    final text = buffer.toString();
    final chunks = split(text, 12);
    expect(chunks, isNotEmpty);
    for (final chunk in chunks) {
      expect(wordTokens(chunk.text), lessThanOrEqualTo(12));
      expect(chunk.text.trim(), isNotEmpty);
    }
    expect(
        segmenter.reassemble(chunks, chunks.map((c) => c.text).toList()), text);
  });

  test('preserves unicode punctuation as sentence boundaries', () {
    const text = 'Première phrase ! Deuxième phrase ? Troisième phrase…';
    final chunks = split(text, 4);
    expect(chunks.length, greaterThan(1));
    expect(
        segmenter.reassemble(chunks, chunks.map((c) => c.text).toList()), text);
  });
}
