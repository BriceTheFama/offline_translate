import 'package:meta/meta.dart';

/// One unit of text handed to the model, plus the whitespace that followed it
/// in the source. Keeping the separator lets the reassembled translation
/// preserve the original paragraph structure.
@immutable
class TextChunk {
  /// Creates a chunk.
  const TextChunk(this.text, this.separator);

  /// The text to translate. Never empty, never pure whitespace.
  final String text;

  /// Whitespace that followed this chunk in the source, e.g. `'\n\n'`.
  final String separator;

  @override
  String toString() => 'TextChunk(${text.length} chars, '
      'sep: ${separator.replaceAll('\n', r'\n')})';
}

/// Splits long text into model-sized chunks along natural boundaries.
///
/// The split is tried in order of decreasing quality: paragraphs, then
/// sentences, then clauses, then words, and only as a last resort inside a
/// word. Whatever separated two chunks in the source is carried along so the
/// reassembled translation keeps the original layout.
class TextSegmenter {
  /// Creates a segmenter.
  const TextSegmenter();

  /// Sentence-final punctuation followed by whitespace, keeping the
  /// punctuation with the sentence it ends. Also treats `»`, `"` and `'`
  /// as part of the sentence when they trail the punctuation.
  static final RegExp _sentenceEnd =
      RegExp(r"""(?<=[.!?\u2026\u3002\uFF01\uFF1F])["\u201d\u00bb'\)\]]*\s+""");

  /// Clause-level fallback punctuation.
  static final RegExp _clauseEnd = RegExp(r'(?<=[,;:—–])\s+');

  /// Runs of newlines, which delimit paragraphs.
  static final RegExp _paragraphBreak = RegExp(r'\n[ \t]*\n[\s]*|\n');

  /// Splits [text] into chunks of at most [maxTokens] tokens, measured with
  /// [countTokens].
  ///
  /// [countTokens] must include whatever special tokens the model appends, so
  /// that a returned chunk is always safe to feed to the encoder.
  List<TextChunk> split(
    String text, {
    required int maxTokens,
    required int Function(String) countTokens,
  }) {
    if (text.trim().isEmpty) return const [];
    final chunks = <TextChunk>[];
    for (final block in _splitKeepingSeparators(text, _paragraphBreak)) {
      if (block.text.trim().isEmpty) {
        // A blank paragraph only contributes layout; fold it into the previous
        // chunk's separator so it survives reassembly.
        if (chunks.isNotEmpty) {
          final last = chunks.removeLast();
          chunks.add(TextChunk(
              last.text, last.separator + block.text + block.separator));
        }
        continue;
      }
      _packBlock(block, maxTokens, countTokens, chunks);
    }
    return chunks;
  }

  /// Joins translated [parts] back together using the chunk separators.
  ///
  /// [parts] must be aligned with the chunks returned by [split].
  String reassemble(List<TextChunk> chunks, List<String> parts) {
    assert(chunks.length == parts.length, 'chunk/part length mismatch');
    final buffer = StringBuffer();
    for (var i = 0; i < chunks.length; i++) {
      buffer
        ..write(parts[i])
        ..write(chunks[i].separator);
    }
    return buffer.toString();
  }

  void _packBlock(
    TextChunk block,
    int maxTokens,
    int Function(String) countTokens,
    List<TextChunk> out,
  ) {
    if (countTokens(block.text) <= maxTokens) {
      out.add(block);
      return;
    }
    final pieces = _splitKeepingSeparators(block.text, _sentenceEnd);
    final packed = _greedyPack(pieces, maxTokens, countTokens, (piece) {
      final clauses = _splitKeepingSeparators(piece.text, _clauseEnd);
      if (clauses.length > 1) {
        return _greedyPack(clauses, maxTokens, countTokens, _splitByWords);
      }
      return _splitByWords(piece);
    });
    if (packed.isEmpty) return;
    // The block's own trailing separator belongs to its last chunk.
    final last = packed.removeLast();
    packed.add(TextChunk(last.text, last.separator + block.separator));
    out.addAll(packed);
  }

  /// Greedily fills chunks up to [maxTokens], delegating pieces that do not fit
  /// on their own to [splitOversized].
  List<TextChunk> _greedyPack(
    List<TextChunk> pieces,
    int maxTokens,
    int Function(String) countTokens,
    List<TextChunk> Function(TextChunk) splitOversized,
  ) {
    final out = <TextChunk>[];
    var buffer = '';
    var bufferSeparator = '';
    void flush() {
      if (buffer.isNotEmpty) {
        out.add(TextChunk(buffer, bufferSeparator));
        buffer = '';
        bufferSeparator = '';
      }
    }

    for (final piece in pieces) {
      if (piece.text.trim().isEmpty) {
        bufferSeparator += piece.text + piece.separator;
        continue;
      }
      if (countTokens(piece.text) > maxTokens) {
        flush();
        final parts = splitOversized(piece);
        if (parts.isNotEmpty) {
          final last = parts.removeLast();
          parts.add(TextChunk(last.text, last.separator + piece.separator));
          out.addAll(parts);
        }
        continue;
      }
      final candidate =
          buffer.isEmpty ? piece.text : '$buffer$bufferSeparator${piece.text}';
      if (countTokens(candidate) <= maxTokens) {
        buffer = candidate;
        bufferSeparator = piece.separator;
      } else {
        flush();
        buffer = piece.text;
        bufferSeparator = piece.separator;
      }
    }
    flush();
    return out;
  }

  /// Last-resort split: on word boundaries, and inside a word if a single word
  /// still exceeds the limit.
  static List<TextChunk> _splitByWords(TextChunk piece) {
    final words = _splitKeepingSeparators(piece.text, RegExp(r'\s+'));
    if (words.length > 1) return words;
    // One enormous "word" (a URL, a base64 blob): cut it into fixed slices.
    const slice = 120;
    final out = <TextChunk>[];
    final text = piece.text;
    for (var i = 0; i < text.length; i += slice) {
      final end = i + slice < text.length ? i + slice : text.length;
      out.add(TextChunk(text.substring(i, end), ''));
    }
    return out;
  }

  /// Splits on [pattern], attaching each match to the piece it followed.
  static List<TextChunk> _splitKeepingSeparators(String text, RegExp pattern) {
    final out = <TextChunk>[];
    var start = 0;
    for (final match in pattern.allMatches(text)) {
      final body = text.substring(start, match.start);
      if (body.isNotEmpty || match.start > start) {
        out.add(TextChunk(body, match.group(0)!));
      } else if (out.isNotEmpty) {
        final last = out.removeLast();
        out.add(TextChunk(last.text, last.separator + match.group(0)!));
      } else {
        out.add(TextChunk('', match.group(0)!));
      }
      start = match.end;
    }
    if (start < text.length) {
      out.add(TextChunk(text.substring(start), ''));
    }
    return out;
  }
}
