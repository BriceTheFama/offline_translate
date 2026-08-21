import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Where this demo gets its model bundles from.
///
/// Two sources are wired up:
///
/// * a **local directory**, used when `--dart-define=OT_MODELS_DIR=/path` is
///   passed (desktop) or when the bundles were pushed to the device's
///   documents directory. This is what makes the airplane-mode demo possible
///   without any server.
/// * an **HTTPS base URL**, used otherwise, given by
///   `--dart-define=OT_MODELS_URL=https://…`.
class DemoModelSources {
  /// Local directory holding `<pair>/manifest.json`, if configured.
  static const String localDir = String.fromEnvironment(
    'OT_MODELS_DIR',
    defaultValue: '',
  );

  /// Remote base URL holding `<pair>/manifest.json`, if configured.
  static const String remoteUrl = String.fromEnvironment(
    'OT_MODELS_URL',
    defaultValue: '',
  );

  /// Resolves the source to use, preferring anything local.
  ///
  /// On Android and iOS the app's documents directory is also probed, so a
  /// bundle can be pushed with `adb push` or the Xcode file browser and picked
  /// up without a rebuild.
  static Future<({ModelSource source, String description})> resolve() async {
    if (localDir.isNotEmpty && Directory(localDir).existsSync()) {
      return (
        source: DirectoryModelSource(localDir),
        description: 'local directory $localDir',
      );
    }
    if (!kIsWeb) {
      final docs = await getApplicationDocumentsDirectory();
      final pushed = Directory(p.join(docs.path, 'ot-models'));
      if (pushed.existsSync()) {
        return (
          source: DirectoryModelSource(pushed.path),
          description: 'device directory ${pushed.path}',
        );
      }
    }
    if (remoteUrl.isNotEmpty) {
      return (
        source: HttpModelSource(baseUrl: Uri.parse(remoteUrl)),
        description: 'remote $remoteUrl',
      );
    }
    return (
      source: const DirectoryModelSource('/nonexistent'),
      description:
          'none configured — pass --dart-define=OT_MODELS_DIR=<path> '
          'or --dart-define=OT_MODELS_URL=<url>',
    );
  }
}
