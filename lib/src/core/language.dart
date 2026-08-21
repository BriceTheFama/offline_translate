import 'package:meta/meta.dart';

/// Languages supported by the built-in OPUS-MT model catalogue.
///
/// The codes are ISO 639-1 and match the `from`/`to` parts of the model
/// identifiers published by the Helsinki-NLP OPUS-MT project.
enum Language {
  /// English.
  en('en', 'English', 'English'),

  /// French.
  fr('fr', 'French', 'Français'),

  /// Spanish.
  es('es', 'Spanish', 'Español'),

  /// German.
  de('de', 'German', 'Deutsch');

  const Language(this.code, this.englishName, this.nativeName);

  /// ISO 639-1 code, e.g. `en`.
  final String code;

  /// English display name, e.g. `French`.
  final String englishName;

  /// Display name in the language itself, e.g. `Français`.
  final String nativeName;

  /// Parses an ISO 639-1 [code] (case insensitive, `fr-FR` style tags are
  /// accepted and reduced to their primary subtag).
  ///
  /// Returns `null` when the code is not supported.
  static Language? tryParse(String code) {
    final primary = code.split(RegExp('[-_]')).first.toLowerCase();
    for (final l in Language.values) {
      if (l.code == primary) return l;
    }
    return null;
  }

  /// Parses an ISO 639-1 [code], throwing [ArgumentError] when unsupported.
  static Language parse(String code) {
    final l = tryParse(code);
    if (l == null) {
      throw ArgumentError.value(code, 'code', 'Unsupported language');
    }
    return l;
  }
}

/// An ordered (source, target) language pair identifying one translation model.
@immutable
class LanguagePair {
  /// Creates a pair translating [from] into [to].
  const LanguagePair(this.from, this.to);

  /// Source language.
  final Language from;

  /// Target language.
  final Language to;

  /// Canonical identifier, e.g. `en-fr`. Used as a directory and cache key.
  String get id => '${from.code}-${to.code}';

  /// The reverse direction, e.g. `en-fr` becomes `fr-en`.
  LanguagePair get reversed => LanguagePair(to, from);

  @override
  bool operator ==(Object other) =>
      other is LanguagePair && other.from == from && other.to == to;

  @override
  int get hashCode => Object.hash(from, to);

  @override
  String toString() => id;
}
