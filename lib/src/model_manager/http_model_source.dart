import 'dart:io';

import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;

import '../core/language.dart';
import '../exceptions/exceptions.dart';
import 'model_source.dart';

/// Fetches model bundles over HTTPS from a static file host.
///
/// The layout expected under [baseUrl] is one directory per direction:
///
/// ```text
/// <baseUrl>/en-fr/manifest.json
/// <baseUrl>/en-fr/encoder.onnx
/// <baseUrl>/en-fr/encoder.data
/// <baseUrl>/en-fr/decoder.onnx
/// <baseUrl>/en-fr/decoder.data
/// <baseUrl>/en-fr/embedding.data
/// <baseUrl>/en-fr/source.spm
/// ```
///
/// Use [HttpModelSource.official] for the bundles published alongside this
/// package, or pass your own [baseUrl] to host them yourself — any static file
/// server works.
///
/// This is the only part of the package that ever touches the network, and it
/// is only reachable through [ModelManager.install]. Nothing here is consulted
/// once a model is on disk, which is why a translator built with no source at
/// all still works.
class HttpModelSource implements ModelSource {
  /// Creates a source rooted at [baseUrl].
  HttpModelSource({required this.baseUrl, http.Client? client})
      : _client = client ?? http.Client(),
        _ownsClient = client == null;

  /// The bundles published for this package, on Hugging Face.
  ///
  /// ```dart
  /// final translator = await OfflineTranslator.initialize(
  ///   languages: {Language.en, Language.fr},
  ///   modelSource: HttpModelSource.official(),
  /// );
  /// await translator.installModel(from: Language.en, to: Language.fr);
  /// ```
  ///
  /// This is *not* the default for [OfflineTranslator.initialize]. Omitting
  /// `modelSource` leaves the translator with no network path at all, and that
  /// stays the default deliberately: an application that ships its own models,
  /// or that has already installed them, should not carry a download path it
  /// never uses. Naming this constructor is how you opt in.
  ///
  /// Hugging Face's URL layout is already the one this package expects, and its
  /// LFS redirects are followed transparently by `package:http`. Pin
  /// [revision] to a commit or tag if you would rather not track `main`.
  factory HttpModelSource.official(
          {String revision = 'main', http.Client? client}) =>
      HttpModelSource(
        baseUrl: Uri.parse('$officialRepository/resolve/$revision'),
        client: client,
      );

  /// Where the published bundles live. Their licence is **MPL-2.0**; see
  /// `doc/licensing.md` before redistributing them yourself.
  static const String officialRepository =
      'https://huggingface.co/fama-corp/offline_translate';

  /// Root URL holding one directory per language direction.
  final Uri baseUrl;

  final http.Client _client;
  final bool _ownsClient;

  Uri _uri(LanguagePair pair, String name) =>
      baseUrl.replace(path: p.url.join(baseUrl.path, pair.id, name));

  @override
  Future<String> fetchManifest(LanguagePair pair) async {
    final uri = _uri(pair, 'manifest.json');
    try {
      final response = await _client.get(uri);
      if (response.statusCode != 200) {
        throw ModelDownloadException(
            pair.id, 'HTTP ${response.statusCode} for $uri');
      }
      return response.body;
    } on OfflineTranslatorException {
      rethrow;
    } catch (e) {
      throw ModelDownloadException(pair.id, 'cannot reach $uri ($e)');
    }
  }

  @override
  Future<void> fetchFile(
    LanguagePair pair,
    String name,
    File destination, {
    void Function(int delta)? onBytes,
  }) async {
    final uri = _uri(pair, name);
    final request = http.Request('GET', uri);
    http.StreamedResponse response;
    try {
      response = await _client.send(request);
    } catch (e) {
      throw ModelDownloadException(pair.id, 'cannot reach $uri ($e)');
    }
    if (response.statusCode != 200) {
      throw ModelDownloadException(
          pair.id, 'HTTP ${response.statusCode} for $uri');
    }
    final sink = destination.openWrite();
    try {
      await for (final chunk in response.stream) {
        sink.add(chunk);
        onBytes?.call(chunk.length);
      }
      await sink.flush();
    } finally {
      await sink.close();
    }
  }

  /// Closes the underlying HTTP client when this source created it.
  void close() {
    if (_ownsClient) _client.close();
  }
}
