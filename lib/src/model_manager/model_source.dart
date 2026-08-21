import 'dart:io';

import '../core/language.dart';

/// Progress of a model installation.
class InstallProgress {
  /// Creates a progress snapshot.
  const InstallProgress({
    required this.pair,
    required this.stage,
    required this.receivedBytes,
    required this.totalBytes,
    this.currentFile,
  });

  /// The model being installed.
  final LanguagePair pair;

  /// What the installer is currently doing.
  final InstallStage stage;

  /// Bytes fetched so far across all files.
  final int receivedBytes;

  /// Expected total size in bytes, or `0` when it is not known yet.
  final int totalBytes;

  /// Name of the file being fetched or verified, when applicable.
  final String? currentFile;

  /// Completion between 0 and 1, or `null` when [totalBytes] is unknown.
  double? get fraction =>
      totalBytes > 0 ? (receivedBytes / totalBytes).clamp(0.0, 1.0) : null;

  @override
  String toString() => 'InstallProgress(${pair.id}, $stage, '
      '$receivedBytes/$totalBytes${currentFile != null ? ', $currentFile' : ''})';
}

/// Stages an installation moves through.
enum InstallStage {
  /// Fetching the manifest that describes the model files.
  manifest,

  /// Downloading or copying model files.
  downloading,

  /// Checking file sizes and SHA-256 checksums.
  verifying,

  /// The model is installed and ready.
  done,
}

/// Where model files come from.
///
/// Implementations must be able to produce a `manifest.json` and every file it
/// lists. Once a model is installed nothing here is ever consulted again, which
/// is what keeps translation fully offline.
abstract class ModelSource {
  /// Fetches the raw `manifest.json` text for [pair].
  Future<String> fetchManifest(LanguagePair pair);

  /// Writes the file [name] of [pair] into [destination].
  ///
  /// [onBytes] is called with the number of bytes appended since the previous
  /// call, so the caller can aggregate progress across files.
  Future<void> fetchFile(
    LanguagePair pair,
    String name,
    File destination, {
    void Function(int delta)? onBytes,
  });
}
