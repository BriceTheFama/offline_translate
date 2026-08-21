import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translate/offline_translate.dart';

void main() {
  group('Language', () {
    test('exposes the four V1 languages with display names', () {
      expect(Language.values.map((l) => l.code).toList(),
          <String>['en', 'fr', 'es', 'de']);
      expect(Language.fr.englishName, 'French');
      expect(Language.fr.nativeName, 'Français');
    });

    test('parses codes case-insensitively and accepts locale tags', () {
      expect(Language.tryParse('fr'), Language.fr);
      expect(Language.tryParse('FR'), Language.fr);
      expect(Language.tryParse('fr-FR'), Language.fr);
      expect(Language.tryParse('de_DE'), Language.de);
      expect(Language.tryParse('pt'), isNull);
      expect(Language.tryParse(''), isNull);
    });

    test('parse throws on unsupported codes', () {
      expect(Language.parse('es'), Language.es);
      expect(() => Language.parse('zz'), throwsArgumentError);
    });
  });

  group('LanguagePair', () {
    test('has a canonical id and a reverse', () {
      const pair = LanguagePair(Language.en, Language.fr);
      expect(pair.id, 'en-fr');
      expect(pair.toString(), 'en-fr');
      expect(pair.reversed, const LanguagePair(Language.fr, Language.en));
    });

    test('is a value type', () {
      expect(const LanguagePair(Language.en, Language.fr),
          const LanguagePair(Language.en, Language.fr));
      expect(const LanguagePair(Language.en, Language.fr).hashCode,
          const LanguagePair(Language.en, Language.fr).hashCode);
      expect(const LanguagePair(Language.en, Language.fr),
          isNot(const LanguagePair(Language.fr, Language.en)));
    });
  });
}
