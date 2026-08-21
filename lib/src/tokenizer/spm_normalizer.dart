import 'dart:convert';
import 'dart:typed_data';

/// SentencePiece text normalizer (`nmt_nfkc` and friends).
///
/// This is a faithful port of `sentencepiece::Normalizer`. It reads the
/// `precompiled_charsmap` blob straight out of the `.spm` model, which is a
/// darts-clone double-array trie mapping byte sequences to their replacement,
/// followed by a NUL-delimited blob holding those replacements.
///
/// Working on UTF-8 bytes (rather than Dart's UTF-16 code units) is what makes
/// the output bit-identical to the reference implementation.
class SpmNormalizer {
  /// Creates a normalizer from a raw `precompiled_charsmap` blob.
  ///
  /// [charsMap] may be empty, in which case only whitespace handling applies.
  factory SpmNormalizer({
    required Uint8List charsMap,
    required bool addDummyPrefix,
    required bool removeExtraWhitespaces,
    required bool escapeWhitespaces,
  }) {
    Uint32List trie = Uint32List(0);
    Uint8List normalized = Uint8List(0);
    if (charsMap.isNotEmpty) {
      final header = ByteData.sublistView(charsMap, 0, 4);
      final trieSize = header.getUint32(0, Endian.little);
      // `Uint32List.sublistView` requires 4-byte alignment; copy to be safe.
      final trieBytes =
          Uint8List.fromList(Uint8List.sublistView(charsMap, 4, 4 + trieSize));
      trie = Uint32List.sublistView(trieBytes);
      normalized = Uint8List.sublistView(charsMap, 4 + trieSize);
    }
    return SpmNormalizer._(
      trie: trie,
      normalizedBlob: normalized,
      addDummyPrefix: addDummyPrefix,
      removeExtraWhitespaces: removeExtraWhitespaces,
      escapeWhitespaces: escapeWhitespaces,
    );
  }

  SpmNormalizer._({
    required Uint32List trie,
    required Uint8List normalizedBlob,
    required this.addDummyPrefix,
    required this.removeExtraWhitespaces,
    required this.escapeWhitespaces,
  })  : _trie = trie,
        _normalized = normalizedBlob;

  final Uint32List _trie;
  final Uint8List _normalized;

  /// Whether a leading space is inserted before normalizing.
  final bool addDummyPrefix;

  /// Whether repeated whitespace collapses to a single space.
  final bool removeExtraWhitespaces;

  /// Whether spaces become `U+2581` (`▁`).
  final bool escapeWhitespaces;

  /// U+2581 LOWER ONE EIGHTH BLOCK, encoded as UTF-8.
  static const List<int> _spaceSymbol = [0xE2, 0x96, 0x81];
  static const List<int> _replacementChar = [0xEF, 0xBF, 0xBD];

  // darts-clone double-array unit accessors.
  static bool _hasLeaf(int u) => ((u >> 8) & 1) == 1;
  static int _value(int u) => u & 0x7FFFFFFF;
  static int _label(int u) => u & 0x800000FF;
  static int _offset(int u) => (u >> 10) << ((u & 0x200) >> 6);

  /// Normalizes [input] and returns the normalized text.
  String normalize(String input) =>
      utf8.decode(normalizeBytes(utf8.encode(input)), allowMalformed: true);

  /// Normalizes UTF-8 [input] and returns the normalized UTF-8 bytes.
  Uint8List normalizeBytes(List<int> input) {
    final out = BytesBuilder(copy: false);
    if (input.isEmpty) return out.takeBytes();

    var pos = 0;
    final end = input.length;

    if (removeExtraWhitespaces) {
      while (pos < end) {
        final p = _normalizePrefix(input, pos, end);
        if (!(p.length == 1 && p.isSpace)) break;
        pos += p.consumed;
      }
    }
    if (pos >= end) return out.takeBytes();

    final space = escapeWhitespaces ? _spaceSymbol : const [0x20];

    if (addDummyPrefix) out.add(space);

    var isPrevSpace = removeExtraWhitespaces;
    while (pos < end) {
      final p = _normalizePrefix(input, pos, end);
      var start = p.start;
      var len = p.length;
      final src = p.source;

      // Drop leading spaces when the previous piece ended with one.
      while (isPrevSpace && len > 0 && src[start] == 0x20) {
        start++;
        len--;
      }

      if (len > 0) {
        for (var n = 0; n < len; n++) {
          final b = src[start + n];
          if (b == 0x20) {
            out.add(space);
          } else {
            out.addByte(b);
          }
        }
        isPrevSpace = src[start + len - 1] == 0x20;
      }

      pos += p.consumed;
      if (!removeExtraWhitespaces) isPrevSpace = false;
    }

    var bytes = out.takeBytes();
    if (removeExtraWhitespaces) {
      var length = bytes.length;
      while (_endsWith(bytes, length, space)) {
        length -= space.length;
      }
      if (length != bytes.length) {
        bytes = Uint8List.sublistView(bytes, 0, length);
      }
    }
    return bytes;
  }

  static bool _endsWith(Uint8List bytes, int length, List<int> suffix) {
    if (length < suffix.length) return false;
    for (var i = 0; i < suffix.length; i++) {
      if (bytes[length - suffix.length + i] != suffix[i]) return false;
    }
    return true;
  }

  _Prefix _normalizePrefix(List<int> input, int pos, int end) {
    var longestLength = 0;
    var longestValue = 0;

    if (_trie.isNotEmpty) {
      var nodePos = 0;
      var unit = _trie[nodePos];
      nodePos ^= _offset(unit);
      for (var i = pos; i < end; i++) {
        final b = input[i];
        nodePos ^= b;
        if (nodePos >= _trie.length) break;
        unit = _trie[nodePos];
        if (_label(unit) != b) break;
        nodePos ^= _offset(unit);
        if (_hasLeaf(unit)) {
          if (nodePos >= _trie.length) break;
          longestLength = i - pos + 1;
          longestValue = _value(_trie[nodePos]);
        }
      }
    }

    if (longestLength == 0 || longestValue >= _normalized.length) {
      final len = _utf8CharLength(input, pos, end);
      if (len == 0) {
        // Malformed UTF-8: emit U+FFFD but consume a single byte.
        return _Prefix(_replacementChar, 0, _replacementChar.length, 1);
      }
      return _Prefix(input, pos, len, len);
    }

    // Replacement strings inside `_normalized` are NUL terminated.
    var stop = longestValue;
    while (stop < _normalized.length && _normalized[stop] != 0) {
      stop++;
    }
    return _Prefix(
        _normalized, longestValue, stop - longestValue, longestLength);
  }

  /// Returns the length in bytes of the UTF-8 sequence at [pos], or 0 when the
  /// bytes are not a valid encoding.
  static int _utf8CharLength(List<int> b, int pos, int end) {
    final c = b[pos];
    int len;
    if (c < 0x80) {
      return 1;
    } else if (c < 0xC2) {
      return 0;
    } else if (c < 0xE0) {
      len = 2;
    } else if (c < 0xF0) {
      len = 3;
    } else if (c < 0xF5) {
      len = 4;
    } else {
      return 0;
    }
    if (pos + len > end) return 0;
    for (var i = 1; i < len; i++) {
      if ((b[pos + i] & 0xC0) != 0x80) return 0;
    }
    return len;
  }
}

class _Prefix {
  const _Prefix(this.source, this.start, this.length, this.consumed);

  final List<int> source;
  final int start;
  final int length;
  final int consumed;

  bool get isSpace => length == 1 && source[start] == 0x20;
}
