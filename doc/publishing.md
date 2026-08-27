# Publishing, and testing on a real device

Two separate jobs. Neither is done yet, and each has a prerequisite that is
easy to miss.

---

## Part 1 — Publishing to pub.dev

### 1.1 Two things must exist first

**A real git repository.** The working tree is not under version control today:

```console
$ git rev-parse --is-inside-work-tree
fatal: not a git repository
```

`pubspec.yaml` already declares `repository:` and `issue_tracker:` pointing at
`github.com/BriceTheFama/offline_translate`. pub.dev does not verify that the
URL resolves, but it links to it from the package page and awards pub points for
it, so pointing at a repository that does not exist is both a lost score and a
broken promise to anyone who clicks it.

```sh
cd ~/perso/offline_translate/offline_translate
git init -b main
git add .
git commit -m "offline_translate 0.3.0"
gh repo create BriceTheFama/offline_translate --public --source=. --push
```

If you pick a different repository name or owner, change `repository:` and
`issue_tracker:` in `pubspec.yaml` to match before publishing.

> The `.pubignore` already keeps `test/fixtures/` (2 MB of tokenizer reference
> data) and `third_party/` (the ONNX Runtime headers) out of the published
> archive, but they belong in git.

**The models must be hosted.** This is the one that actually blocks usefulness:
the package ships no weights, so until the bundles are reachable, anyone who
adds `offline_translate` to their `pubspec.yaml` gets a translator that cannot
translate. See Part 2 — do it *before* publishing.

### 1.2 Publish

```sh
cd ~/perso/offline_translate/offline_translate
flutter pub publish --dry-run     # currently: 0 warnings, 357 KB
flutter pub publish
```

`flutter pub publish` opens a browser for Google sign-in and uploads under that
account. Two things worth knowing before you press it:

* **Publishing is effectively permanent.** A version can be *retracted* within
  7 days, which hides it from new resolutions but does not delete it, and the
  package **name is claimed forever**. That is why `offline_translator` is now
  unavailable to you — someone claimed it in October 2025 for something else.
* **Consider a verified publisher.** If you own a domain, creating a publisher
  on pub.dev and setting `publish_to` accordingly shows a verified badge instead
  of a bare account. It can be added later, but moving an existing package to a
  publisher is extra work.

### 1.3 Check the score before and after

pub.dev grades on conventions this package already meets — analysis, formatting,
documentation, example, dependency freshness, platform support. Run the same
analysis locally first:

```sh
dart pub global activate pana
dart pub global run pana --no-warning .
```

The one thing pana will flag that is deliberate: `ffigen` is a dev dependency
pinned below its newest major, because the generated bindings are checked in and
regenerating them requires the pinned ONNX Runtime header. That is a considered
choice, documented in `doc/onnx-runtime.md`.

---

## Part 2 — Hosting the model bundles

1.27 GB across 12 directions, 102-120 MB each. The package needs
`<base>/<pair>/<file>`, nothing more.

### 2.1 Hugging Face (recommended)

Its URL layout is already the one `HttpModelSource` expects, and redirects to
its LFS CDN are followed transparently — verified against a real object:

```console
$ # what HttpModelSource.fetchFile does, against huggingface.co
status: 200
streamed bytes: 778395     # exactly source.spm
```

The published bundles live at
[`fama-corp/offline_translate`](https://huggingface.co/fama-corp/offline_translate),
which is what `HttpModelSource.official()` resolves to. To publish there, or to
your own repository:

```sh
pip install huggingface_hub
hf auth login

python3 tool/upload_models.py ~/ot-models-tiny --dry-run
python3 tool/upload_models.py ~/ot-models-tiny
```

The script creates the repository if needed, writes a model card listing every
direction with its size, family, vocabulary, upstream project and **licence** —
all read from the manifests, so the card cannot drift from what the repository
actually contains — writes the canonical MPL-2.0 text to `LICENSE` when any
bundle needs it, and uploads only the files a bundle needs.

Consumers then write:

```dart
final translator = await OfflineTranslator.initialize(
  languages: {Language.en, Language.fr},
  modelSource: HttpModelSource.official(),
);
```

Verify the result end to end, from the published URL to a translation:

```sh
flutter test tool/verify_published_test.dart
```

Hugging Face is also the honest place for these files: right next to the
checkpoints they were converted from.

**MPL-2.0 means the upload is a redistribution.** Whoever hosts converted
bundles must keep them under MPL-2.0 and make the source form available — the
conversion pipeline (`tool/build_tiny_model.py`, `tool/marian_binary.py`,
`tool/tiny_transformer.py`) plus the upstream checkpoint. The generated model
card points at this repository for exactly that reason, so **this repository has
to be public** for the obligation to be met.

### 2.2 Alternatives

| | Fit | Notes |
|---|---|---|
| **Cloudflare R2** | exact | No egress fees. The right answer if you want control and a custom domain. |
| **S3 + CloudFront** | exact | Standard, egress costs money. |
| **GitHub Releases** | works | One release per direction, tagged exactly `en-fr`, `fr-en`, … so that `<base>/<pair>/<file>` resolves. Base URL is `https://github.com/OWNER/REPO/releases/download`. 2 GB per-file limit is not a problem here. |
| **Bundled in the app** | no host | Ship the bundles in the IPA/APK or an expansion file and use `DirectoryModelSource`. Adds ~104 MB per direction to the download, but needs no network permission at all. |

Once the URL is settled, put it in the README so the first example a reader
sees actually runs.

---

## Part 3 — Testing on a real Android device

Everything already works over USB; `adb reverse` behaves the same on a phone as
on the emulator.

```sh
# 1. On the phone: Settings > About > tap "Build number" 7 times,
#    then Developer options > USB debugging. Plug in, accept the prompt.
adb devices          # confirm it is listed as "device", not "unauthorized"

# 2. Functional suite (debug — this is the only mode `flutter test` builds)
cd example
python3 -m http.server 8099 --bind 127.0.0.1 --directory ~/ot-models &
adb reverse tcp:8099 tcp:8099
flutter test integration_test/translation_test.dart    -d <serial> --dart-define=OT_MODELS_URL=http://127.0.0.1:8099
flutter test integration_test/all_directions_test.dart -d <serial> --dart-define=OT_MODELS_URL=http://127.0.0.1:8099
flutter test integration_test/stability_test.dart      -d <serial> --dart-define=OT_MODELS_URL=http://127.0.0.1:8099
```

### 3.1 Get numbers that mean something

`flutter test integration_test` **only builds in debug**, which inflates every
Dart-side cost — the tokenizer especially. For figures that reflect a shipped
app, drive the demo in profile mode instead:

```sh
tool/device_bench.sh <serial>              # profile build, one command
tool/device_bench.sh <serial> ~/ot-models de-es
```

It prints the device model, ABI, core count and memory, then the benchmark.
Measured on the 4-core emulator as a reference point:

```text
bench_mode=profile cores=4 rss=625MB
bench_hello=32.3ms chunks=1
bench_sentence20=223.6ms chunks=1
bench_words100=1734.2ms chunks=1
bench_words500=2743.9ms chunks=2
bench_leak_200x=-8MB rss=654MB
bench_async_doc=35868ms chars=13258 chunks=30 ticks=4469 worst_stall=44ms
APK size (arm64, profile): 112.6 MB
```

Profile mode is 15-25 % faster than debug on the longer inputs, and the worker
isolate looks even better: the worst UI stall during a 13 000-character document
drops from 116 ms to **44 ms**.

### 3.2 Prove it is offline

```sh
tool/offline_proof.sh <serial>
```

Installs the model with the network up, then switches Wi-Fi and mobile data off,
removes the adb tunnel, stops the host server, force-stops and relaunches the
app, and asserts it still translates and reports `online=false`.

---

## Part 4 — Testing on a real iPhone

Two things need doing that the simulator did not require.

### 4.1 Signing, and the bundle identifier

The example still uses Apple's placeholder identifier:

```text
example/ios/Runner.xcodeproj  PRODUCT_BUNDLE_IDENTIFIER = com.example.offlineTranslatorExample
```

`com.example.*` is heavily used and Apple will often refuse to provision it
("bundle identifier is not available"). Change it to something you own:

```sh
open example/ios/Runner.xcworkspace
# Runner target > Signing & Capabilities:
#   Team: your Apple ID (a free account works, 7-day provisioning)
#   Bundle Identifier: com.yourname.offlinetranslate.example
```

A free Apple ID is enough for on-device debugging; the profile expires after
7 days and the app stops launching until you rebuild.

### 4.2 Getting the models onto the phone

There is no `adb reverse` for iOS. The example is now set up for the simplest
route — no server, no cleartext-HTTP exception:

```xml
<!-- example/ios/Runner/Info.plist, already added -->
<key>UIFileSharingEnabled</key><true/>
<key>LSSupportsOpeningDocumentsInPlace</key><true/>
```

1. Build and run once so the app exists on the phone.
2. Open **Finder → your iPhone → Files → offline_translator_example**.
3. Drag a folder named `ot-models` containing `en-fr/`, `fr-en/`, … into it.

The demo probes the documents directory for `ot-models/` automatically, so no
`--dart-define` is needed. This also exercises `DirectoryModelSource`, which is
what an application shipping its own models would use.

The alternative is to serve the bundles from your Mac over Wi-Fi and point
`OT_MODELS_URL` at the Mac's LAN address — but plain HTTP then needs an App
Transport Security exception in `Info.plist`, which is worth avoiding for a
demo.

### 4.3 Run

```sh
cd example
flutter devices                                  # find the iPhone's id
flutter test integration_test/translation_test.dart    -d <iphone-id>
flutter test integration_test/all_directions_test.dart -d <iphone-id>

# profile-mode numbers
flutter run --profile --dart-define=OT_AUTORUN=bench -d <iphone-id>
# read the OT_AUTORUN lines in the console
```

For the offline check on iOS, put the phone in Airplane Mode, force-quit the app
and relaunch it: `online=false`, `installed_before=true`, and the translations
still appear.

---

## What has not been verified here

Every number in `doc/performance.md` comes from an **emulator, a simulator, or
macOS**. The 4-core Android emulator is the closest proxy to a mid-range phone,
but it is not one:

* it runs on the host CPU, with host memory bandwidth and no thermal limit;
* nothing throttles after 30 seconds of sustained decoding, which a real phone
  will do on a long document;
* battery cost has not been measured at all.

Expect a real mid-range phone to be **slower than the emulator numbers**, and
expect sustained translation of a long document to throttle. Re-run
`tool/device_bench.sh` on the target hardware before quoting anything to users,
and add the results to `doc/performance.md` alongside the emulator figures
rather than replacing them.
