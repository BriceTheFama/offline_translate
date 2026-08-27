import 'dart:typed_data';

import 'spm_model.dart';

/// One segmented piece and the SentencePiece id it came from.
class SpmToken {
  /// Creates a token.
  const SpmToken(this.piece, this.id, {required this.isUnknown});

  /// The surface form of the piece, e.g. `▁the`.
  final String piece;

  /// SentencePiece id, or the model's unknown id.
  final int id;

  /// Whether the piece could not be covered by the vocabulary.
  final bool isUnknown;

  @override
  String toString() => piece;
}

/// SentencePiece Unigram segmenter.
///
/// Port of `sentencepiece::unigram::Model::EncodeOptimized` followed by the
/// unknown-run merging performed by `SentencePieceProcessor`. Scores are
/// accumulated in 32-bit precision so that near-ties resolve exactly the way
/// the reference C++ implementation resolves them.
class UnigramSegmenter {
  /// Builds a segmenter from a parsed `.spm` [model].
  factory UnigramSegmenter(SpmModel model) {
    final index = <String, int>{};
    var maxLength = 1;
    var minScore = double.infinity;
    var unknownId = 0;
    for (var id = 0; id < model.pieces.length; id++) {
      final type = model.types[id];
      if (type == SpmPieceType.unknown) {
        unknownId = id;
        continue;
      }
      // `Model::Model` puts NORMAL, USER_DEFINED and UNUSED pieces in the
      // trie and keeps CONTROL, UNKNOWN and BYTE out of it. Byte pieces matter
      // here: a byte-fallback vocabulary carries 256 of them, and letting them
      // be matched as text would segment a literal "<0x41>" into one token.
      if (type == SpmPieceType.control || type == SpmPieceType.byte) continue;
      final piece = model.pieces[id];
      if (piece.isEmpty) continue;
      index[piece] = id;
      final len = piece.runes.length;
      if (len > maxLength) maxLength = len;
      if (type == SpmPieceType.normal && model.scores[id] < minScore) {
        minScore = model.scores[id];
      }
    }
    if (minScore == double.infinity) minScore = 0;
    return UnigramSegmenter._(
      index: index,
      scores: model.scores,
      maxPieceLength: maxLength,
      minScore: minScore,
      unknownId: unknownId,
    );
  }

  UnigramSegmenter._({
    required Map<String, int> index,
    required Float32List scores,
    required this.maxPieceLength,
    required this.minScore,
    required this.unknownId,
  })  : _index = index,
        _scores = scores;

  final Map<String, int> _index;
  final Float32List _scores;

  /// Longest vocabulary piece, in Unicode code points.
  final int maxPieceLength;

  /// Lowest score across normal pieces; unknown nodes score below it.
  final double minScore;

  /// SentencePiece id of `<unk>`.
  final int unknownId;

  /// Penalty applied to unknown single-character nodes.
  static const double unknownPenalty = 10.0;

  /// Threshold at which accumulated scores are re-based, mirroring
  /// `kScoreResetThreshold` in the reference encoder.
  static const double _scoreResetThreshold = 100000.0;

  /// Segments already-normalized [text] into vocabulary pieces.
  List<String> segment(String text) =>
      segmentTokens(text).map((t) => t.piece).toList(growable: false);

  /// Segments already-normalized [text], keeping ids and unknown flags.
  List<SpmToken> segmentTokens(String text) {
    if (text.isEmpty) return const [];

    // Code point boundaries expressed as UTF-16 indices.
    final bounds = <int>[];
    for (var i = 0; i < text.length;) {
      bounds.add(i);
      final unit = text.codeUnitAt(i);
      i += (unit >= 0xD800 && unit <= 0xDBFF && i + 1 < text.length) ? 2 : 1;
    }
    bounds.add(text.length);
    final n = bounds.length - 1; // number of characters

    final unknownScore = minScore - unknownPenalty;
    // Single-element view used to round intermediate sums to 32-bit precision,
    // matching the `float` arithmetic of the reference encoder.
    final round = Float32List(1);
    final best = Float32List(n + 1);
    final startsAt = Int32List(n + 1)..fillRange(0, n + 1, -1);
    final ids = Int32List(n + 1)..fillRange(0, n + 1, -1);

    var maxFrontier = 0;
    for (var i = 0; i < n; i++) {
      var scoreHere = best[i];
      if (scoreHere < -_scoreResetThreshold ||
          scoreHere > _scoreResetThreshold) {
        for (var k = i; k <= maxFrontier; k++) {
          if (k == i || startsAt[k] != -1) best[k] = best[k] - scoreHere;
        }
        scoreHere = 0;
      }

      final start = bounds[i];
      var hasSingleChar = false;
      final limit = i + maxPieceLength < n ? i + maxPieceLength : n;
      for (var j = i + 1; j <= limit; j++) {
        final id = _index[text.substring(start, bounds[j])];
        if (id == null) continue;
        if (j > maxFrontier) maxFrontier = j;
        round[0] = _scores[id] + scoreHere;
        final candidate = round[0];
        if (startsAt[j] == -1 || candidate > best[j]) {
          best[j] = candidate;
          startsAt[j] = i;
          ids[j] = id;
        }
        if (j == i + 1) hasSingleChar = true;
      }

      if (!hasSingleChar) {
        final j = i + 1;
        if (j > maxFrontier) maxFrontier = j;
        round[0] = unknownScore + scoreHere;
        final candidate = round[0];
        if (startsAt[j] == -1 || candidate > best[j]) {
          best[j] = candidate;
          startsAt[j] = i;
          ids[j] = unknownId;
        }
      }
    }

    // Backtrack, merging continuous runs of unknown pieces.
    final reversed = <SpmToken>[];
    var end = n;
    while (end > 0) {
      final from = startsAt[end];
      if (from < 0) break;
      final id = ids[end];
      final isUnknown = id == unknownId;
      final piece = text.substring(bounds[from], bounds[end]);
      if (isUnknown && reversed.isNotEmpty && reversed.last.isUnknown) {
        final next = reversed.removeLast();
        reversed.add(SpmToken(piece + next.piece, unknownId, isUnknown: true));
      } else {
        reversed.add(SpmToken(piece, id, isUnknown: isUnknown));
      }
      end = from;
    }
    return reversed.reversed.toList(growable: false);
  }
}
