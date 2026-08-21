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
/// <baseUrl>/en-fr/decoder.onnx
/// <baseUrl>/en-fr/source.spm
/// <baseUrl>/en-fr/vocab.json
/// ```
///
/// This is the only part of the package that ever touches the network, and it
/// is only reachable through [ModelManager.install].
class HttpModelSource implements ModelSource {
  /// Creates a source rooted at [baseUrl].
  HttpModelSource({required this.baseUrl, http.Client? client})
      : _client = client ?? http.Client(),
        _ownsClient = client == null;

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
