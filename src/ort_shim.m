// Keeps ONNX Runtime alive in the link.
//
// The `onnxruntime-c` pod vendors a *static* xcframework, and everything that
// calls it does so from Dart through dart:ffi at runtime. No Objective-C or
// Swift in a Flutter application references ONNX Runtime, so the linker would
// discard every member of the archive as unreachable and
// `DynamicLibrary.process()` would find no symbols.
//
// CocoaPods already puts `-ObjC` in the application's OTHER_LDFLAGS, which
// forces the linker to load any archive member that defines an Objective-C
// class. Defining one here is therefore enough: this object file gets pulled
// in, its reference to `OrtGetApiBase` is resolved, and ONNX Runtime comes
// with it. Nothing calls the class — the reference is the whole point.

#import <Foundation/Foundation.h>

#if __has_include(<onnxruntime/onnxruntime_c_api.h>)
#import <onnxruntime/onnxruntime_c_api.h>
#else
#include "onnxruntime_c_api.h"
#endif

@interface OfflineTranslatorOrtKeepAlive : NSObject
@end

@implementation OfflineTranslatorOrtKeepAlive

+ (const void *)apiBase {
  return (const void *)OrtGetApiBase();
}

@end
