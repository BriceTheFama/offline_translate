import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Where this demo gets its model bundles from.
///
/// The default needs no configuration at all: it downloads from the bundles
/// published for the package, so `flutter run` on a phone works with nothing
/// but a network connection for the first launch. The overrides exist for
/// development and for the airplane-mode demo:
///
/// * `--dart-define=OT_MODELS_DIR=/path` — a local directory, on desktop;
/// * a `ot-models/` directory pushed into the app's documents directory, which
///   is how a bundle reaches a phone without a server (`adb push`, or the Xcode
///   file browser);
/// * `--dart-define=OT_MODELS_URL=https://…` — your own static host;
/// * otherwise, [HttpModelSource.official].
///
/// The order matters: anything local wins, so once a bundle is on the device
/// the demo stops depending on a network even for a fresh install.
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
        description: remoteUrl,
      );
    }
    return (
      source: HttpModelSource.official(),
      description:
          'published bundles '
          '(${Uri.parse(HttpModelSource.officialRepository).pathSegments.join('/')})',
    );
  }
}
