import 'package:flutter_test/flutter_test.dart';
import 'package:offline_translate/offline_translate.dart';

void main() {
  const enFr = LanguagePair(Language.en, Language.fr);
  const frEn = LanguagePair(Language.fr, Language.en);

  test('stores and returns a translation', () {
    final cache = TranslationCache()..put(enFr, 'Hello', 'Bonjour');
    expect(cache.get(enFr, 'Hello'), 'Bonjour');
    expect(cache.length, 1);
  });

  test('is keyed by direction as well as text', () {
    final cache = TranslationCache()..put(enFr, 'Hello', 'Bonjour');
    expect(cache.get(frEn, 'Hello'), isNull);
  });

  test('misses on text that was never stored', () {
    final cache = TranslationCache()..put(enFr, 'Hello', 'Bonjour');
    expect(cache.get(enFr, 'hello'), isNull);
    expect(cache.get(enFr, 'Hello '), isNull);
  });

  test('evicts the least recently used entry past the limit', () {
    final cache = TranslationCache(maxEntries: 2)
      ..put(enFr, 'a', '1')
      ..put(enFr, 'b', '2');
    // Touch 'a' so 'b' becomes the least recently used.
    expect(cache.get(enFr, 'a'), '1');
    cache.put(enFr, 'c', '3');
    expect(cache.length, 2);
    expect(cache.get(enFr, 'b'), isNull);
    expect(cache.get(enFr, 'a'), '1');
    expect(cache.get(enFr, 'c'), '3');
  });

  test('overwrites an existing entry without growing', () {
    final cache = TranslationCache(maxEntries: 2)
      ..put(enFr, 'a', '1')
      ..put(enFr, 'a', '2');
    expect(cache.length, 1);
    expect(cache.get(enFr, 'a'), '2');
  });

  test('expires entries past the time to live', () async {
    final cache = TranslationCache(
      timeToLive: const Duration(milliseconds: 30),
    )..put(enFr, 'Hello', 'Bonjour');
    expect(cache.get(enFr, 'Hello'), 'Bonjour');
    await Future<void>.delayed(const Duration(milliseconds: 60));
    expect(cache.get(enFr, 'Hello'), isNull);
    cache.purgeExpired();
    expect(cache.isEmpty, isTrue);
  });

  test('clears everything or one direction', () {
    final cache = TranslationCache()
      ..put(enFr, 'Hello', 'Bonjour')
      ..put(frEn, 'Bonjour', 'Hello');
    cache.clear(pair: enFr);
    expect(cache.get(enFr, 'Hello'), isNull);
    expect(cache.get(frEn, 'Bonjour'), 'Hello');
    cache.clear();
    expect(cache.isEmpty, isTrue);
  });

  test('rejects a non-positive size', () {
    expect(
        () => TranslationCache(maxEntries: 0), throwsA(isA<AssertionError>()));
  });
}
