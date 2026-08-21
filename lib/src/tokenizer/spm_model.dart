import 'dart:convert';
import 'dart:typed_data';

/// SentencePiece piece types, mirroring `ModelProto.SentencePiece.Type`.
class SpmPieceType {
  /// Normal, trainable piece.
  static const int normal = 1;

  /// The unknown piece (`<unk>`).
  static const int unknown = 2;

  /// Control symbol (`<s>`, `</s>`), never produced by the segmenter.
  static const int control = 3;

  /// User defined symbol, always matched greedily.
  static const int userDefined = 4;

  /// Byte fallback piece.
  static const int byte = 6;

  /// Piece present in the model but excluded from segmentation.
  static const int unused = 5;
}

/// A minimal reader for the `sentencepiece.ModelProto` wire format.
///
/// Only the fields required for inference are decoded: the piece list
/// (`pieces`) and the `normalizer_spec`. This lets the package consume the
/// original, unmodified `source.spm` files published with the OPUS-MT models,
/// so users can verify provenance against the upstream checksums.
class SpmModel {
  SpmModel._({
    required this.pieces,
    required this.scores,
    required this.types,
    required this.precompiledCharsMap,
    required this.addDummyPrefix,
    required this.removeExtraWhitespaces,
    required this.escapeWhitespaces,
  });

  /// Piece strings, indexed by SentencePiece id.
  final List<String> pieces;

  /// Log-probability of every piece, indexed by SentencePiece id.
  final Float32List scores;

  /// Piece type, see [SpmPieceType].
  final Uint8List types;

  /// Raw `precompiled_charsmap` blob used by the normalizer (may be empty).
  final Uint8List precompiledCharsMap;

  /// Whether a leading space is added before normalization.
  final bool addDummyPrefix;

  /// Whether runs of whitespace collapse into a single space.
  final bool removeExtraWhitespaces;

  /// Whether spaces are replaced with `U+2581`.
  final bool escapeWhitespaces;

  /// Parses a `.spm` model file.
  static SpmModel parse(Uint8List bytes) {
    final pieces = <String>[];
    final scores = <double>[];
    final types = <int>[];
    Uint8List charsMap = Uint8List(0);
    var addDummyPrefix = true;
    var removeExtraWhitespaces = true;
    var escapeWhitespaces = true;

    final r = _ProtoReader(bytes);
    while (!r.atEnd) {
      final tag = r.readVarint();
      final field = tag >> 3;
      final wire = tag & 7;
      if (field == 1 && wire == 2) {
        // repeated SentencePiece pieces = 1;
        final sub = r.readLengthDelimited();
        final p = _ProtoReader(sub);
        var piece = '';
        var score = 0.0;
        var type = SpmPieceType.normal;
        while (!p.atEnd) {
          final t = p.readVarint();
          switch (t >> 3) {
            case 1:
              piece = utf8.decode(p.readLengthDelimited());
            case 2:
              score = p.readFloat32();
            case 3:
              type = p.readVarint();
            default:
              p.skip(t & 7);
          }
        }
        pieces.add(piece);
        scores.add(score);
        types.add(type);
      } else if (field == 3 && wire == 2) {
        // optional NormalizerSpec normalizer_spec = 3;
        final sub = r.readLengthDelimited();
        final n = _ProtoReader(sub);
        while (!n.atEnd) {
          final t = n.readVarint();
          switch (t >> 3) {
            case 2:
              charsMap = n.readLengthDelimited();
            case 3:
              addDummyPrefix = n.readVarint() != 0;
            case 4:
              removeExtraWhitespaces = n.readVarint() != 0;
            case 5:
              escapeWhitespaces = n.readVarint() != 0;
            default:
              n.skip(t & 7);
          }
        }
      } else {
        r.skip(wire);
      }
    }

    return SpmModel._(
      pieces: pieces,
      scores: Float32List.fromList(scores),
      types: Uint8List.fromList(types),
      precompiledCharsMap: charsMap,
      addDummyPrefix: addDummyPrefix,
      removeExtraWhitespaces: removeExtraWhitespaces,
      escapeWhitespaces: escapeWhitespaces,
    );
  }
}

class _ProtoReader {
  _ProtoReader(this._bytes) : _view = ByteData.sublistView(_bytes);

  final Uint8List _bytes;
  final ByteData _view;
  int _pos = 0;

  bool get atEnd => _pos >= _bytes.length;

  int readVarint() {
    var result = 0;
    var shift = 0;
    while (true) {
      final b = _bytes[_pos++];
      result |= (b & 0x7f) << shift;
      if (b & 0x80 == 0) return result;
      shift += 7;
    }
  }

  Uint8List readLengthDelimited() {
    final len = readVarint();
    final out = Uint8List.sublistView(_bytes, _pos, _pos + len);
    _pos += len;
    return out;
  }

  double readFloat32() {
    final v = _view.getFloat32(_pos, Endian.little);
    _pos += 4;
    return v;
  }

  void skip(int wireType) {
    switch (wireType) {
      case 0:
        readVarint();
      case 1:
        _pos += 8;
      case 2:
        // Note: `_pos += readVarint()` would be wrong, the compound assignment
        // captures `_pos` before readVarint() advances it.
        final length = readVarint();
        _pos += length;
      case 5:
        _pos += 4;
      default:
        throw FormatException(
            'Unsupported protobuf wire type $wireType at offset $_pos');
    }
  }
}
