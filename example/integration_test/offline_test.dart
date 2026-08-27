import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:offline_translate/offline_translate.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

/// Two-phase proof that translation survives losing the network.
///
/// Phase 1 installs the model into a directory that is *not* cleaned up:
///
/// ```sh
/// adb reverse tcp:8099 tcp:8099
/// flutter test integration_test/offline_test.dart -d <device> \
///   --dart-define=OT_PHASE=install --dart-define=OT_MODELS_URL=http://127.0.0.1:8099
/// ```
///
/// Then the network is taken away entirely — Wi-Fi and mobile data off, the
/// adb reverse tunnel removed, the host server stopped — the app is rebuilt and
/// relaunched, and phase 2 translates with no source configured at all:
///
/// ```sh
/// adb shell svc wifi disable && adb shell svc data disable
/// adb reverse --remove-all
/// flutter test integration_test/offline_test.dart -d <device> \
///   --dart-define=OT_PHASE=translate
/// ```
///
/// Phase 2 deliberately configures a source pointing at an unreachable host, so
/// that any accidental network dependency in the translation path fails loudly
/// instead of silently succeeding.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  const phase = String.fromEnvironment('OT_PHASE', defaultValue: 'install');
  const remoteUrl = String.fromEnvironment('OT_MODELS_URL', defaultValue: '');
  const localDir = String.fromEnvironment('OT_MODELS_DIR', defaultValue: '');

  Future<String> persistentRoot() async {
    final support = await getApplicationSupportDirectory();
    final dir = Directory(p.join(support.path, 'ot-offline-proof'));
    if (!dir.existsSync()) dir.createSync(recursive: true);
    return dir.path;
  }

  testWidgets(
    'phase 1: install the model while the network is up',
    (tester) async {
      if (phase != 'install') return;
      final ModelSource source;
      if (localDir.isNotEmpty && Directory(localDir).existsSync()) {
        source = DirectoryModelSource(localDir);
      } else if (remoteUrl.isNotEmpty) {
        source = HttpModelSource(baseUrl: Uri.parse(remoteUrl));
      } else {
        fail('phase 1 needs OT_MODELS_URL or OT_MODELS_DIR');
      }

      final manager = FileModelManager(
        source: source,
        rootPath: await persistentRoot(),
      );
      final translator = await OfflineTranslator.initialize(
        modelManager: manager,
      );
      final info = await translator.installModel(
        from: Language.en,
        to: Language.fr,
      );
      expect(info.id, 'en-fr');
      await manager.verify(const LanguagePair(Language.en, Language.fr));

      await translator.preload(from: Language.en, to: Language.fr);
      final result = translator.translate(
        'Hello, how are you?',
        from: Language.en,
        to: Language.fr,
      );
      expect(result.translatedText, 'Bonjour, comment allez-vous ?');
      // ignore: avoid_print
      print('PHASE1 installed to ${info.path}: "${result.translatedText}"');
      await translator.dispose();
    },
    timeout: const Timeout(Duration(minutes: 10)),
  );

  testWidgets(
    'phase 2: translate with the network gone',
    (tester) async {
      if (phase != 'translate') return;

      // A source that cannot possibly resolve. Nothing in the translation path
      // should ever consult it.
      final manager = FileModelManager(
        source: HttpModelSource(
          baseUrl: Uri.parse('https://offline-translator.invalid/models'),
        ),
        rootPath: await persistentRoot(),
      );

      final installed = await manager.installedModels();
      expect(
        installed.map((m) => m.id),
        contains('en-fr'),
        reason: 'phase 1 must run first, on the same device',
      );

      // Checksums still match after the app was killed and relaunched.
      await manager.verify(const LanguagePair(Language.en, Language.fr));

      final translator = await OfflineTranslator.initialize(
        languages: const {Language.en, Language.fr},
        defaultLanguage: Language.fr,
        modelManager: manager,
      );

      // Prove the network really is unreachable from this process right now.
      var networkIsDown = false;
      try {
        final socket = await Socket.connect(
          'example.com',
          80,
          timeout: const Duration(seconds: 5),
        );
        socket.destroy();
      } catch (_) {
        networkIsDown = true;
      }

      final short = translator.translate(
        'Hello, how are you?',
        from: Language.en,
        to: Language.fr,
      );
      expect(short.translatedText, 'Bonjour, comment allez-vous ?');

      final long = await translator.translateLong(
        'The network is switched off. This sentence is being translated '
        'entirely on the device.\n\n'
        'Nothing leaves the phone, and no server is involved at any point.',
        from: Language.en,
        to: Language.fr,
      );
      expect(long.translatedText.trim(), isNotEmpty);
      expect(long.chunkCount, greaterThanOrEqualTo(2));

      // ignore: avoid_print
      print(
        'PHASE2 network_down=$networkIsDown\n'
        '  sync : ${short.translatedText}\n'
        '  async: ${long.translatedText.replaceAll("\n", " | ")}',
      );
      expect(
        networkIsDown,
        isTrue,
        reason: 'the device still had network; disable it before phase 2',
      );

      await translator.dispose();
    },
    timeout: const Timeout(Duration(minutes: 10)),
  );
}
