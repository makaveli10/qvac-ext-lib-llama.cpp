plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.compose.compiler)
}

@Suppress("UnstableApiUsage")
android {
    namespace = "ai.ggml.llamacpp"
    compileSdk = 36

    defaultConfig {
        applicationId = "ai.ggml.llamacpp"
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    defaultConfig {
        externalNativeBuild {
            cmake {
                arguments(
                    "-DGGML_OPENMP=OFF",
                    "-DGGML_LLAMAFILE=OFF",
                    "-DLLAMA_CURL=OFF",
                    "-DGGML_VULKAN=1",
                    "-DLLAMA_BUILD_TESTS=1", // test need tools
                    "-DLLAMA_BUILD_TOOLS=1",
                    "-DLLAMA_BUILD_EXAMPLES=1",
                    "-DLLAMA_BUILD_COMMON=ON",
                    "-DCMAKE_BUILD_TYPE=Release"
                )

                abiFilters("arm64-v8a")
            }
        }
    }

    buildFeatures {
        viewBinding = true
        compose = true
    }
}

dependencies {

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.ui)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material)
    implementation(libs.androidx.material.icons.core)
    implementation(libs.androidx.material.icons.extended)
    implementation(libs.androidx.foundation.layout)
    implementation(libs.androidx.navigation.safe.args.generator) {
        exclude(group = "xpp3", module = "xpp3")
    }
    implementation(libs.androidx.navigation.compose)
    implementation(libs.androidx.runtime.saveable)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.runtime)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    debugImplementation(libs.androidx.ui.tooling)
}
