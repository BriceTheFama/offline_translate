import 'package:flutter/material.dart';

import 'autorun.dart';
import 'translator_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  if (autorunEnabled) {
    // Headless self-check mode, see tool/offline_proof.sh.
    runApp(const _AutorunApp());
    return;
  }
  runApp(const OfflineTranslatorDemo());
}

/// Demo application for the `offline_translate` package.
class OfflineTranslatorDemo extends StatelessWidget {
  /// Creates the demo app.
  const OfflineTranslatorDemo({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Offline Translator',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF3F6FD8),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorSchemeSeed: const Color(0xFF3F6FD8),
        brightness: Brightness.dark,
        useMaterial3: true,
      ),
      home: const TranslatorPage(),
    );
  }
}

class _AutorunApp extends StatefulWidget {
  const _AutorunApp();

  @override
  State<_AutorunApp> createState() => _AutorunAppState();
}

class _AutorunAppState extends State<_AutorunApp> {
  List<String> _lines = const <String>['running...'];

  @override
  void initState() {
    super.initState();
    runSelfCheck().then((report) {
      if (mounted) setState(() => _lines = report.lines);
    });
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        body: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: ListView(
              children: <Widget>[
                for (final line in _lines)
                  Text(line, style: const TextStyle(fontSize: 11)),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
