plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.offline_translator_example"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_17.toString()
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.example.offline_translator_example"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    // A debug APK carries `libonnxruntime.so` and `libflutter.so` for **every**
    // ABI — around 190 MB of native libraries — which is enough to fail
    // installation on an emulator with a few hundred megabytes free, with a
    // misleading "not enough space". Flutter's `--target-platform` trims its
    // own libraries but not a plugin's, because those arrive as prebuilt
    // `jniLibs`, and `ndk.abiFilters` does not touch prebuilt libraries either.
    // Excluding the unwanted ABIs at packaging time does.
    //
    //     flutter build apk --debug --android-project-arg=abi=arm64-v8a
    //     echo 'abi=arm64-v8a' >> android/gradle.properties   # for flutter test
    //
    // Release builds with `--split-per-abi` are already one ABI each and need
    // none of this.
    (project.findProperty("abi") as String?)?.let { requested ->
        val keep = requested.split(",").map(String::trim).toSet()
        packaging {
            jniLibs {
                listOf("arm64-v8a", "armeabi-v7a", "x86", "x86_64")
                    .filterNot(keep::contains)
                    .forEach { excludes += "lib/$it/**" }
            }
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}

flutter {
    source = "../.."
}
