import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translator_example/main.dart';

void main() {
  testWidgets('renders the translator page', (tester) async {
    await tester.pumpWidget(const OfflineTranslatorDemo());
    await tester.pump();
    expect(find.text('Offline Translator'), findsOneWidget);
  });
}
