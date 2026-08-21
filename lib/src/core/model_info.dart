import 'dart:convert';

import 'package:meta/meta.dart';

import 'language.dart';

/// Shape and special-token constants of a Marian encoder-decoder checkpoint.
///
/// These come from the checkpoint's `config.json` and are frozen into the
/// model manifest at build time so the engine never has to guess.
@immutable
class ModelArchitecture {
  /// Creates an architecture description.
  const ModelArchitecture({
    required this.decoderLayers,
    required this.decoderAttentionHeads,
    required this.headDimension,
    required this.decoderStartTokenId,
    required this.eosTokenId,
    required this.padTokenId,
    required this.maxPositionEmbeddings,
    required this.vocabSize,
  });

  /// Number of decoder layers, i.e. the number of KV-cache entries.
  final int decoderLayers;

  /// Number of decoder attention heads.
  final int decoderAttentionHeads;

  /// Size of a single attention head (`d_model / heads`).
  final int headDimension;

  /// Token the decoder is primed with (Marian uses `<pad>`).
  final int decoderStartTokenId;

  /// End-of-sequence token that stops generation.
  final int eosTokenId;

  /// Padding token.
  final int padTokenId;

  /// Number of positions the model was trained with.
  final int maxPositionEmbeddings;

  /// Size of the shared vocabulary.
  final int vocabSize;

  /// Parses an architecture block from a manifest.
  factory ModelArchitecture.fromJson(Map<String, dynamic> json) =>
      ModelArchitecture(
        decoderLayers: json['decoder_layers'] as int,
        decoderAttentionHeads: json['decoder_attention_heads'] as int,
        headDimension: json['head_dimension'] as int,
        decoderStartTokenId: json['decoder_start_token_id'] as int,
        eosTokenId: json['eos_token_id'] as int,
        padTokenId: json['pad_token_id'] as int,
        maxPositionEmbeddings: json['max_position_embeddings'] as int,
        vocabSize: json['vocab_size'] as int,
      );

  /// Serialises the architecture block.
  Map<String, dynamic> toJson() => <String, dynamic>{
        'decoder_layers': decoderLayers,
        'decoder_attention_heads': decoderAttentionHeads,
        'head_dimension': headDimension,
        'decoder_start_token_id': decoderStartTokenId,
        'eos_token_id': eosTokenId,
        'pad_token_id': padTokenId,
        'max_position_embeddings': maxPositionEmbeddings,
        'vocab_size': vocabSize,
      };
}

/// One file belonging to a model, with its integrity metadata.
@immutable
class ModelFile {
  /// Creates a file entry.
  const ModelFile({
    required this.name,
    required this.size,
    required this.sha256,
  });

  /// File name, relative to the model directory.
  final String name;

  /// Size in bytes.
  final int size;

  /// Lowercase hex SHA-256 of the file contents.
  final String sha256;

  /// Parses a file entry.
  factory ModelFile.fromJson(Map<String, dynamic> json) => ModelFile(
        name: json['name'] as String,
        size: json['size'] as int,
        sha256: json['sha256'] as String,
      );

  /// Serialises the file entry.
  Map<String, dynamic> toJson() =>
      <String, dynamic>{'name': name, 'size': size, 'sha256': sha256};
}

/// Everything the package knows about one translation model.
@immutable
class ModelInfo {
  /// Creates model metadata.
  const ModelInfo({
    required this.from,
    required this.to,
    required this.version,
    required this.size,
    required this.checksum,
    required this.path,
    required this.architecture,
    required this.files,
    required this.baseModel,
    required this.license,
    this.quantization = 'int8',
  });

  /// Source language.
  final Language from;

  /// Target language.
  final Language to;

  /// Model package version, e.g. `1.0.0`.
  final String version;

  /// Total size of all model files in bytes.
  final int size;

  /// SHA-256 over the sorted per-file checksums; identifies the whole bundle.
  final String checksum;

  /// Absolute path of the directory holding the model, or an empty string for
  /// catalogue entries that are not installed yet.
  final String path;

  /// Shape and special-token constants.
  final ModelArchitecture architecture;

  /// The files making up this model.
  final List<ModelFile> files;

  /// Upstream checkpoint this model was converted from.
  final String baseModel;

  /// SPDX identifier of the upstream checkpoint's license.
  final String license;

  /// Weight precision, e.g. `int8` or `fp32`.
  final String quantization;

  /// The direction this model translates.
  LanguagePair get pair => LanguagePair(from, to);

  /// Canonical identifier, e.g. `en-fr`.
  String get id => pair.id;

  /// Parses a `manifest.json`. [path] is filled in by the model manager.
  factory ModelInfo.fromJson(Map<String, dynamic> json, {String path = ''}) {
    final files = (json['files'] as List<dynamic>)
        .map((dynamic e) => ModelFile.fromJson(e as Map<String, dynamic>))
        .toList(growable: false);
    return ModelInfo(
      from: Language.parse(json['from'] as String),
      to: Language.parse(json['to'] as String),
      version: json['version'] as String,
      size: files.fold<int>(0, (a, f) => a + f.size),
      checksum: json['checksum'] as String,
      path: path,
      architecture: ModelArchitecture.fromJson(
          json['architecture'] as Map<String, dynamic>),
      files: files,
      baseModel: json['base_model'] as String,
      license: json['license'] as String,
      quantization: json['quantization'] as String? ?? 'int8',
    );
  }

  /// Parses a manifest from its JSON text.
  static ModelInfo parse(String source, {String path = ''}) =>
      ModelInfo.fromJson(jsonDecode(source) as Map<String, dynamic>,
          path: path);

  /// Serialises the manifest.
  Map<String, dynamic> toJson() => <String, dynamic>{
        'from': from.code,
        'to': to.code,
        'version': version,
        'checksum': checksum,
        'base_model': baseModel,
        'license': license,
        'quantization': quantization,
        'architecture': architecture.toJson(),
        'files': files.map((f) => f.toJson()).toList(),
      };

  /// Serialises the manifest to JSON text, for sending to a worker isolate.
  String toJsonString() => jsonEncode(toJson());

  /// Returns a copy with [path] replaced.
  ModelInfo withPath(String newPath) => ModelInfo(
        from: from,
        to: to,
        version: version,
        size: size,
        checksum: checksum,
        path: newPath,
        architecture: architecture,
        files: files,
        baseModel: baseModel,
        license: license,
        quantization: quantization,
      );

  @override
  String toString() =>
      'ModelInfo($id, v$version, ${(size / 1048576).toStringAsFixed(1)} MB, '
      '$quantization)';
}
