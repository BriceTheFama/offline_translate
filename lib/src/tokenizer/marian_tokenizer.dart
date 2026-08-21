import 'dart:convert';
import 'dart:typed_data';

import 'spm_model.dart';
import 'spm_normalizer.dart';
import 'unigram_model.dart';

/// Tokenizer for MarianMT / OPUS-MT models.
///
/// Matches `transformers.MarianTokenizer`: text is normalized and segmented
/// with the *source* SentencePiece model, then pieces are mapped to ids
/// through the shared `vocab.json`, and `</s>` is appended.
///
/// Decoding only needs the shared vocabulary: SentencePiece pieces are
/// concatenated and `▁` is turned back into a space.
class MarianTokenizer {
  MarianTokenizer._({
    required SpmNormalizer normalizer,
    required UnigramSegmenter segmenter,
    required Map<String, int> vocab,
    required List<String> reverseVocab,
    required this.unknownId,
    required this.eosId,
    required this.padId,
  })  : _normalizer = normalizer,
        _segmenter = segmenter,
        _vocab = vocab,
        _reverseVocab = reverseVocab;

  final SpmNormalizer _normalizer;
  final UnigramSegmenter _segmenter;
  final Map<String, int> _vocab;
  final List<String> _reverseVocab;

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

  /// Builds a tokenizer from the raw `source.spm` and `vocab.json` contents.
  static MarianTokenizer fromAssets({
    required Uint8List spmBytes,
    required String vocabJson,
  }) {
    final model = SpmModel.parse(spmBytes);
    final decoded = jsonDecode(vocabJson) as Map<String, dynamic>;
    final vocab = <String, int>{};
    var maxId = -1;
    decoded.forEach((k, dynamic v) {
      final id = v as int;
      vocab[k] = id;
      if (id > maxId) maxId = id;
    });
    final reverse = List<String>.filled(maxId + 1, '');
    vocab.forEach((k, v) => reverse[v] = k);

    return MarianTokenizer._(
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

  /// Normalizes and segments [text] into SentencePiece pieces.
  List<String> tokenize(String text) {
    var body = text;
    final match = _languageCodePattern.matchAsPrefix(text);
    final prefix = match?.group(0);
    if (prefix != null) body = text.substring(prefix.length);
    final pieces = _segmenter.segment(_normalizer.normalize(body));
    if (prefix == null) return pieces;
    return <String>[prefix, ...pieces];
  }

  /// Encodes [text] into model input ids, terminated by `</s>`.
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
    for (final id in ids) {
      if (skipSpecialTokens &&
          (id == eosId || id == padId || id == unknownId)) {
        continue;
      }
      buffer.write(idToPiece(id));
    }
    final text = buffer.toString().replaceAll('▁', ' ');
    return text.startsWith(' ') ? text.substring(1) : text;
  }
}
