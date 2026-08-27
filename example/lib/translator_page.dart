import 'dart:async';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:offline_translate/offline_translate.dart';

import 'model_setup.dart';

const String _longSample = '''
Machine translation has changed a great deal over the last decade. Early
systems relied on hand written rules and large phrase tables, and they produced
text that was often understandable but rarely natural.

Neural models replaced those pipelines with a single network trained end to
end. The encoder reads the source sentence and turns it into a sequence of
vectors, and the decoder produces the target sentence one token at a time,
attending to the encoder output at every step.

Running such a model on a phone used to be out of the question. Quantisation,
better runtimes and smaller architectures have made it practical: a distilled
translation model now fits in about thirty megabytes and produces a sentence in
a few dozen milliseconds, with no server involved and no data leaving the
device.
''';

/// The demo screen: pick a direction, install the model, translate.
class TranslatorPage extends StatefulWidget {
  /// Creates the page.
  const TranslatorPage({super.key});

  @override
  State<TranslatorPage> createState() => _TranslatorPageState();
}

class _TranslatorPageState extends State<TranslatorPage> {
  final TextEditingController _input = TextEditingController(
    text: 'Hello, how are you?',
  );

  OfflineTranslator? _translator;
  String _sourceDescription = '';
  Language _from = Language.en;
  Language _to = Language.fr;

  bool _busy = true;
  String _status = 'Starting…';
  String? _error;
  ModelInfo? _model;
  InstallProgress? _progress;
  TranslationResult? _result;
  String _streamed = '';
  bool? _online;
  String _runtime = '';

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _input.dispose();
    final t = _translator;
    if (t != null) unawaited(t.dispose());
    super.dispose();
  }

  Future<void> _bootstrap() async {
    try {
      final resolved = await DemoModelSources.resolve();
      // Every language the pickers offer has to be declared, or choosing one
      // raises UnsupportedLanguageException. A real application declares only
      // what it needs — that is the point of the parameter — and then never
      // downloads the rest.
      final translator = await OfflineTranslator.initialize(
        languages: Language.values.toSet(),
        defaultLanguage: Language.fr,
        modelSource: resolved.source,
        cache: TranslationCache(maxEntries: 128),
      );
      if (!mounted) return;
      setState(() {
        _translator = translator;
        _sourceDescription = resolved.description;
        _runtime = OfflineTranslator.onnxRuntimeVersion;
      });
      unawaited(_checkNetwork());
      await _refreshModel();
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  LanguagePair get _pair => LanguagePair(_from, _to);

  Future<void> _refreshModel() async {
    final translator = _translator;
    if (translator == null) return;
    final info = await translator.modelManager.getModel(_pair);
    if (!mounted) return;
    setState(() {
      _model = info;
      _status = info == null
          ? 'Model ${_pair.id} is not installed'
          : 'Model ${_pair.id} installed (${_mb(info.size)})';
    });
    if (info != null) {
      setState(() => _busy = true);
      try {
        await translator.preload(from: _from, to: _to);
        if (mounted) setState(() => _status = 'Model ${_pair.id} loaded');
      } catch (e) {
        if (mounted) setState(() => _error = '$e');
      } finally {
        if (mounted) setState(() => _busy = false);
      }
    }
  }

  static String _mb(int bytes) => '${(bytes / 1048576).toStringAsFixed(1)} MB';

  /// Is the network actually reachable right now?
  ///
  /// This is here so the offline demo is self-evident: install a model, turn on
  /// airplane mode, tap refresh, watch this go red — and then translate anyway.
  Future<void> _checkNetwork() async {
    setState(() => _online = null);
    var reachable = false;
    try {
      final socket = await Socket.connect(
        'example.com',
        443,
        timeout: const Duration(seconds: 3),
      );
      socket.destroy();
      reachable = true;
    } catch (_) {
      reachable = false;
    }
    if (mounted) setState(() => _online = reachable);
  }

  Future<void> _install() async {
    final translator = _translator;
    if (translator == null) return;
    setState(() {
      _busy = true;
      _error = null;
      _status = 'Installing ${_pair.id}…';
    });
    try {
      await translator.installModel(
        from: _from,
        to: _to,
        onProgress: (progress) {
          if (mounted) setState(() => _progress = progress);
        },
      );
      if (mounted) setState(() => _progress = null);
      await _refreshModel();
    } catch (e) {
      if (mounted) {
        setState(
          () => _error = e is ModelDownloadException
              ? '$e\n\nNot every direction is published yet. Build one with '
                    '`python3 tool/build_tiny_model.py --pair ${_pair.id}` and host '
                    'it, or pick a direction that is available.'
              : '$e',
        );
      }
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          _progress = null;
        });
      }
    }
  }

  Future<void> _delete() async {
    final translator = _translator;
    if (translator == null) return;
    setState(() => _busy = true);
    try {
      await translator.deleteModel(from: _from, to: _to);
      await _refreshModel();
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  void _swap() {
    setState(() {
      final from = _from;
      _from = _to;
      _to = from;
      _result = null;
      _streamed = '';
    });
    unawaited(_refreshModel());
  }

  /// Runs the synchronous API straight on the UI isolate — the point of the
  /// demo is that a short sentence does not visibly stall the frame.
  void _translateShort() {
    final translator = _translator;
    if (translator == null) return;
    setState(() {
      _error = null;
      _streamed = '';
    });
    try {
      final result = translator.translate(_input.text, from: _from, to: _to);
      setState(() => _result = result);
    } catch (e) {
      setState(() => _error = '$e');
    }
  }

  Future<void> _translateAsync() async {
    final translator = _translator;
    if (translator == null) return;
    setState(() {
      _busy = true;
      _error = null;
      _streamed = '';
    });
    try {
      final result = await translator.translateLong(
        _input.text,
        from: _from,
        to: _to,
      );
      if (mounted) setState(() => _result = result);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _translateStream() async {
    final translator = _translator;
    if (translator == null) return;
    setState(() {
      _busy = true;
      _error = null;
      _result = null;
      _streamed = '';
    });
    final watch = Stopwatch()..start();
    try {
      await for (final chunk in translator.translateStream(
        _input.text,
        from: _from,
        to: _to,
      )) {
        if (!mounted) return;
        setState(() => _streamed += chunk.translatedText);
      }
      if (mounted) {
        setState(() => _status = 'Streamed in ${watch.elapsedMilliseconds} ms');
      }
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final loaded = _translator?.loadedModels.contains(_pair) ?? false;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Offline Translator'),
        bottom: _busy
            ? const PreferredSize(
                preferredSize: Size.fromHeight(3),
                child: LinearProgressIndicator(minHeight: 3),
              )
            : null,
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(16),
          children: <Widget>[
            Row(
              children: <Widget>[
                Expanded(
                  child: _LanguageDropdown(
                    label: 'From',
                    value: _from,
                    onChanged: (v) {
                      setState(() => _from = v);
                      unawaited(_refreshModel());
                    },
                  ),
                ),
                IconButton(
                  onPressed: _swap,
                  icon: const Icon(Icons.swap_horiz),
                  tooltip: 'Swap',
                ),
                Expanded(
                  child: _LanguageDropdown(
                    label: 'To',
                    value: _to,
                    onChanged: (v) {
                      setState(() => _to = v);
                      unawaited(_refreshModel());
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _input,
              maxLines: 6,
              minLines: 3,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                labelText: 'Text to translate',
              ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: <Widget>[
                FilledButton.icon(
                  onPressed: loaded ? _translateShort : null,
                  icon: const Icon(Icons.bolt),
                  label: const Text('translate'),
                ),
                FilledButton.tonalIcon(
                  onPressed: _busy ? null : _translateAsync,
                  icon: const Icon(Icons.notes),
                  label: const Text('translateLong'),
                ),
                OutlinedButton.icon(
                  onPressed: _busy ? null : _translateStream,
                  icon: const Icon(Icons.stream),
                  label: const Text('translateStream'),
                ),
                TextButton(
                  onPressed: () => setState(() {
                    _input.text = _longSample.trim();
                    _result = null;
                    _streamed = '';
                  }),
                  child: const Text('Load long sample'),
                ),
              ],
            ),
            const Divider(height: 32),
            if (_error != null)
              Card(
                color: theme.colorScheme.errorContainer,
                child: Padding(
                  padding: const EdgeInsets.all(12),
                  child: Text(
                    _error!,
                    style: TextStyle(color: theme.colorScheme.onErrorContainer),
                  ),
                ),
              ),
            if (_streamed.isNotEmpty)
              _OutputCard(title: 'Streaming output', body: _streamed),
            if (_result != null)
              _OutputCard(
                title: 'Translation',
                body: _result!.translatedText,
                footer:
                    '${_result!.duration.inMilliseconds} ms · '
                    '${_result!.chunkCount} chunk(s)'
                    '${_result!.fromCache ? ' · from cache' : ''}'
                    '${_result!.truncated ? ' · truncated' : ''}',
              ),
            const SizedBox(height: 16),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Row(
                      children: <Widget>[
                        Text('Model', style: theme.textTheme.titleMedium),
                        const Spacer(),
                        _NetworkChip(online: _online),
                        IconButton(
                          onPressed: _checkNetwork,
                          icon: const Icon(Icons.refresh, size: 18),
                          tooltip: 'Re-check the network',
                          visualDensity: VisualDensity.compact,
                        ),
                      ],
                    ),
                    Text(_status),
                    Text(
                      'Source: $_sourceDescription',
                      style: theme.textTheme.bodySmall,
                    ),
                    if (_model != null) ...<Widget>[
                      Text(
                        '${_model!.baseModel} · ${_model!.license} · '
                        '${_model!.quantization} · '
                        '${_model!.architecture.family.name}',
                        style: theme.textTheme.bodySmall,
                      ),
                      Text(
                        '${_mb(_model!.size)} on disk · '
                        '${_model!.architecture.decoderLayers} decoder layers · '
                        'vocab ${_model!.architecture.vocabSize}',
                        style: theme.textTheme.bodySmall,
                      ),
                    ],
                    Text(
                      '${Platform.operatingSystem} · '
                      '${Platform.numberOfProcessors} cores · '
                      'ONNX Runtime $_runtime',
                      style: theme.textTheme.bodySmall,
                    ),
                    if (_progress != null) ...<Widget>[
                      const SizedBox(height: 8),
                      LinearProgressIndicator(value: _progress!.fraction),
                      Text(
                        '${_progress!.stage.name} '
                        '${_progress!.currentFile ?? ''} '
                        '${_mb(_progress!.receivedBytes)} / '
                        '${_mb(_progress!.totalBytes)}',
                        style: theme.textTheme.bodySmall,
                      ),
                    ],
                    const SizedBox(height: 8),
                    Wrap(
                      spacing: 8,
                      children: <Widget>[
                        FilledButton.tonal(
                          onPressed: _busy ? null : _install,
                          child: const Text('Install model'),
                        ),
                        OutlinedButton(
                          onPressed: _busy || _model == null ? null : _delete,
                          child: const Text('Delete model'),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

/// Whether the network is reachable *right now*, which is the whole point of
/// the offline demo: install a model, switch the device to airplane mode, watch
/// this turn red, and translate anyway.
class _NetworkChip extends StatelessWidget {
  const _NetworkChip({required this.online});

  final bool? online;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    if (online == null) {
      return const SizedBox(
        width: 16,
        height: 16,
        child: CircularProgressIndicator(strokeWidth: 2),
      );
    }
    final offline = !online!;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(
        color: offline
            ? theme.colorScheme.tertiaryContainer
            : theme.colorScheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        offline ? 'OFFLINE' : 'online',
        style: theme.textTheme.labelSmall?.copyWith(
          color: offline
              ? theme.colorScheme.onTertiaryContainer
              : theme.colorScheme.onSurfaceVariant,
          fontWeight: offline ? FontWeight.bold : FontWeight.normal,
        ),
      ),
    );
  }
}

class _LanguageDropdown extends StatelessWidget {
  const _LanguageDropdown({
    required this.label,
    required this.value,
    required this.onChanged,
  });

  final String label;
  final Language value;
  final ValueChanged<Language> onChanged;

  @override
  Widget build(BuildContext context) {
    return DropdownButtonFormField<Language>(
      initialValue: value,
      decoration: InputDecoration(
        labelText: label,
        border: const OutlineInputBorder(),
      ),
      items: <DropdownMenuItem<Language>>[
        for (final language in Language.values)
          DropdownMenuItem<Language>(
            value: language,
            child: Text('${language.nativeName} (${language.code})'),
          ),
      ],
      onChanged: (v) {
        if (v != null) onChanged(v);
      },
    );
  }
}

class _OutputCard extends StatelessWidget {
  const _OutputCard({required this.title, required this.body, this.footer});

  final String title;
  final String body;
  final String? footer;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(title, style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            SelectableText(body, style: theme.textTheme.bodyLarge),
            if (footer != null) ...<Widget>[
              const SizedBox(height: 8),
              Text(footer!, style: theme.textTheme.bodySmall),
            ],
          ],
        ),
      ),
    );
  }
}
