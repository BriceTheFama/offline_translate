import 'dart:collection';

import '../core/language.dart';

/// An optional least-recently-used cache for completed translations.
///
/// Disabled by default. When enabled it is keyed by (direction, exact source
/// text), so it only ever returns a translation the engine itself produced for
/// the identical input.
class TranslationCache {
  /// Creates a cache holding at most [maxEntries] entries, each valid for
  /// [timeToLive] when that is set.
  TranslationCache({this.maxEntries = 256, this.timeToLive})
      : assert(maxEntries > 0, 'maxEntries must be positive');

  /// Maximum number of entries kept; the least recently used entry is evicted.
  final int maxEntries;

  /// How long an entry stays valid. `null` means entries never expire.
  final Duration? timeToLive;

  final LinkedHashMap<String, _Entry> _entries =
      LinkedHashMap<String, _Entry>();

  /// Number of entries currently held.
  int get length => _entries.length;

  /// Whether the cache holds no entries.
  bool get isEmpty => _entries.isEmpty;

  static String _key(LanguagePair pair, String text) => '${pair.id} $text';

  /// Returns the cached translation for [text], or `null`.
  String? get(LanguagePair pair, String text) {
    final key = _key(pair, text);
    final entry = _entries.remove(key);
    if (entry == null) return null;
    final ttl = timeToLive;
    if (ttl != null && DateTime.now().difference(entry.storedAt) > ttl) {
      return null;
    }
    // Reinserting moves the entry to the most-recently-used end.
    _entries[key] = entry;
    return entry.value;
  }

  /// Stores [translation] for [text].
  void put(LanguagePair pair, String text, String translation) {
    final key = _key(pair, text);
    _entries
      ..remove(key)
      ..[key] = _Entry(translation, DateTime.now());
    while (_entries.length > maxEntries) {
      _entries.remove(_entries.keys.first);
    }
  }

  /// Drops every entry, or only those for [pair] when it is given.
  void clear({LanguagePair? pair}) {
    if (pair == null) {
      _entries.clear();
      return;
    }
    final prefix = '${pair.id} ';
    _entries.removeWhere((key, _) => key.startsWith(prefix));
  }

  /// Removes entries that have outlived [timeToLive].
  void purgeExpired() {
    final ttl = timeToLive;
    if (ttl == null) return;
    final now = DateTime.now();
    _entries.removeWhere((_, e) => now.difference(e.storedAt) > ttl);
  }
}

class _Entry {
  _Entry(this.value, this.storedAt);

  final String value;
  final DateTime storedAt;
}
