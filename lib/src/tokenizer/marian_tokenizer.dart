import 'dart:convert';
import 'dart:typed_data';

import 'spm_model.dart';
import 'spm_normalizer.dart';
import 'unigram_model.dart';

/// Tokenizer for Marian models, in both the shapes this package ships.
///
/// Matches `transformers.MarianTokenizer`: text is normalized and segmented
/// with the *source* SentencePiece model, then pieces are mapped to ids
/// through the shared `vocab.json`, and `</s>` is appended.
///
/// Two vocabularies are supported, because the two model families disagree
/// about where ids come from. OPUS-MT segments with a 32 000-piece `source.spm`
/// and then maps those pieces through a separate 59 514-entry `vocab.json`
/// covering both languages. The Firefox Translations students have no second
/// mapping at all: the SentencePiece ids *are* the model ids, so passing no
/// `vocabJson` builds the identity vocabulary straight from the `.spm`.
///
/// Decoding concatenates pieces and turns `▁` back into a space, reassembling
/// any `<0xNN>` byte pieces into UTF-8 on the way.
class MarianTokenizer {
  MarianTokenizer._({
    required SpmNormalizer normalizer,
    required UnigramSegmenter segmenter,
    required Map<String, int> vocab,
    required List<String> reverseVocab,
    required this.unknownId,
    required this.eosId,
    required this.padId,
    required Int32List byteIds,
    required Map<int, int> bytePieceIds,
  })  : _normalizer = normalizer,
        _segmenter = segmenter,
        _vocab = vocab,
        _reverseVocab = reverseVocab,
        _byteIds = byteIds,
        _bytePieceIds = bytePieceIds;

  final SpmNormalizer _normalizer;
  final UnigramSegmenter _segmenter;
  final Map<String, int> _vocab;
  final List<String> _reverseVocab;

  /// Vocabulary id of `<0xNN>` for each byte value, or -1 when the model has no
  /// byte-fallback piece for it. Empty when the model does not use byte
  /// fallback at all.
  final Int32List _byteIds;

  /// Reverse of [_byteIds]: vocabulary id -> byte value, for the decoder.
  final Map<int, int> _bytePieceIds;

  /// Whether unknown characters are emitted as `<0xNN>` byte pieces.
  bool get byteFallback => _byteIds.isNotEmpty;

  /// Id of `<unk>`.
  final int unknownId;

  /// Id of `</s>`, appended to every encoded sequence.
  final int eosId;

  /// Id of `<pad>`, also used as `decoder_start_token_id` by Marian.
  final int padId;

  /// Number of entries in the shared vocabulary.
  int get vocabSize => _reverseVocab.length;

  /// Matches the `>>xx<<` target-language prefix used by multilingual OPUS-MT
  /// checkpoints. Such a prefix is emitted verbatim as its own token.
  static final RegExp _languageCodePattern = RegExp(r'^>>[a-zA-Z_]{2,}<<');

  /// Builds a tokenizer from the raw `source.spm` contents, and optionally the
  /// separate `vocab.json` that OPUS-MT models map their pieces through.
  ///
  /// When [vocabJson] is null the SentencePiece ids are used directly.
  static MarianTokenizer fromAssets({
    required Uint8List spmBytes,
    String? vocabJson,
  }) {
    final model = SpmModel.parse(spmBytes);
    final vocab = <String, int>{};
    var maxId = -1;
    if (vocabJson != null) {
      final decoded = jsonDecode(vocabJson) as Map<String, dynamic>;
      decoded.forEach((k, dynamic v) {
        final id = v as int;
        vocab[k] = id;
        if (id > maxId) maxId = id;
      });
    } else {
      for (var id = 0; id < model.pieces.length; id++) {
        vocab[model.pieces[id]] = id;
      }
      maxId = model.pieces.length - 1;
    }
    final reverse = List<String>.filled(maxId + 1, '');
    vocab.forEach((k, v) {
      if (v >= 0 && v <= maxId) reverse[v] = k;
    });

    // Byte-fallback pieces are declared by the SentencePiece model, but their
    // ids come from whichever vocabulary is in force.
    final byteIds = Int32List(model.byteFallback ? 256 : 0);
    final bytePieceIds = <int, int>{};
    if (model.byteFallback) {
      byteIds.fillRange(0, 256, -1);
      for (var id = 0; id < model.pieces.length; id++) {
        if (model.types[id] != SpmPieceType.byte) continue;
        final value = _parseBytePiece(model.pieces[id]);
        final mapped = vocab[model.pieces[id]];
        if (value == null || mapped == null) continue;
        byteIds[value] = mapped;
        bytePieceIds[mapped] = value;
      }
    }

    return MarianTokenizer._(
      byteIds: byteIds,
      bytePieceIds: bytePieceIds,
      normalizer: SpmNormalizer(
        charsMap: model.precompiledCharsMap,
        addDummyPrefix: model.addDummyPrefix,
        removeExtraWhitespaces: model.removeExtraWhitespaces,
        escapeWhitespaces: model.escapeWhitespaces,
      ),
      segmenter: UnigramSegmenter(model),
      vocab: vocab,
      reverseVocab: reverse,
      unknownId: vocab['<unk>'] ?? 1,
      eosId: vocab['</s>'] ?? 0,
      padId: vocab['<pad>'] ?? (maxId),
    );
  }

  /// `<0x1F>` -> 31. Returns null for anything else.
  static int? _parseBytePiece(String piece) {
    if (piece.length != 6 || !piece.startsWith('<0x') || !piece.endsWith('>')) {
      return null;
    }
    return int.tryParse(piece.substring(3, 5), radix: 16);
  }

  /// Normalizes and segments [text] into SentencePiece pieces.
  ///
  /// With a byte-fallback vocabulary this is what `SentencePieceProcessor`
  /// itself returns: a character the model has never seen appears as one
  /// `<0xNN>` piece per UTF-8 byte, not as a single `<unk>`.
  List<String> tokenize(String text) {
    final pieces = <String>[];
    for (final token in tokenizeDetailed(text)) {
      if (!token.isUnknown || !byteFallback) {
        pieces.add(token.piece);
        continue;
      }
      for (final byte in utf8.encode(token.piece)) {
        final id = _byteIds[byte];
        pieces.add(id >= 0 ? idToPiece(id) : token.piece);
      }
    }
    return pieces;
  }

  /// [tokenize], keeping the unknown flag each piece was segmented with.
  List<SpmToken> tokenizeDetailed(String text) {
    var body = text;
    final match = _languageCodePattern.matchAsPrefix(text);
    final prefix = match?.group(0);
    if (prefix != null) body = text.substring(prefix.length);
    final tokens = _segmenter.segmentTokens(_normalizer.normalize(body));
    if (prefix == null) return tokens;
    return <SpmToken>[
      SpmToken(prefix, _vocab[prefix] ?? unknownId, isUnknown: false),
      ...tokens,
    ];
  }

  /// Encodes [text] into model input ids, terminated by `</s>`.
  ///
  /// With a byte-fallback vocabulary a character the model has never seen
  /// becomes one `<0xNN>` id per UTF-8 byte instead of a single `<unk>`, which
  /// is what `SentencePieceProcessor::Encode` does and what lets the model
  /// copy such a character through to its output.
  Int32List encode(String text) {
    final pieces = tokenize(text);
    final ids = Int32List(pieces.length + 1);
    for (var i = 0; i < pieces.length; i++) {
      ids[i] = _vocab[pieces[i]] ?? unknownId;
    }
    ids[pieces.length] = eosId;
    return ids;
  }

  /// Converts a single piece to its vocabulary id, or [unknownId].
  int pieceToId(String piece) => _vocab[piece] ?? unknownId;

  /// Converts a vocabulary [id] back to its piece.
  String idToPiece(int id) =>
      id >= 0 && id < _reverseVocab.length ? _reverseVocab[id] : '';

  /// Decodes generated [ids] back to text.
  ///
  /// Special tokens (`</s>`, `<pad>`, `<unk>`) are dropped unless
  /// [skipSpecialTokens] is `false`.
  String decode(Iterable<int> ids, {bool skipSpecialTokens = true}) {
    final buffer = StringBuffer();
    final pending = <int>[];
    void flush() {
      if (pending.isEmpty) return;
      buffer.write(utf8.decode(pending, allowMalformed: true));
      pending.clear();
    }

    for (final id in ids) {
      if (skipSpecialTokens &&
          (id == eosId || id == padId || id == unknownId)) {
        continue;
      }
      // Byte pieces are only meaningful as a run: one `<0xNN>` is a third of an
      // accented character, so they are buffered and decoded together.
      final byte = _bytePieceIds[id];
      if (byte != null) {
        pending.add(byte);
        continue;
      }
      flush();
      buffer.write(idToPiece(id));
    }
    flush();
    final text = buffer.toString().replaceAll('▁', ' ');
    return text.startsWith(' ') ? text.substring(1) : text;
  }
}
