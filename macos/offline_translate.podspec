Pod::Spec.new do |s|
  s.name             = 'offline_translate'
  s.version          = '0.6.0'
  s.summary          = 'Offline neural machine translation for Flutter.'
  s.description      = <<-DESC
Runs compact distilled translation models locally with ONNX Runtime.
                       DESC
  s.homepage         = 'https://github.com/BriceTheFama/offline_translate'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'offline_translate' => 'noreply@example.com' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'FlutterMacOS'

  # ONNX Runtime's official C pod. It vendors a static xcframework, so the
  # symbols land in the host application binary and Dart reaches them with
  # DynamicLibrary.process(). Classes/ort_shim.m holds the reference that stops
  # the linker from dead-stripping the archive; see the comment in that file.
  s.dependency 'onnxruntime-c', '1.29.0'

  # onnxruntime-c vendors a *static* xcframework. A pod with `use_frameworks!`
  # cannot depend on one unless it is itself a static framework, so this must
  # stay set — without it `pod install` fails with "transitive dependencies
  # that include statically linked binaries".
  s.static_framework = true

  # onnxruntime-c 1.29.0 declares osx 14.0 as its minimum.
  s.platform = :osx, '14.0'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
  }
end
