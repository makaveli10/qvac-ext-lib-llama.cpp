cmake_minimum_required(VERSION 3.22)

project("llama-android")

# build native_app_glue as a static lib
add_library(native_app_glue STATIC
        ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c)

add_subdirectory(../../../../../ build-llama)

# JNI library
add_library(llama_cpp_android SHARED
        llama-app.cpp)

target_link_libraries(llama_cpp_android
        llama
        common
        android
        log)


