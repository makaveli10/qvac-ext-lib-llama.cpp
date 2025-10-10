set(TARGET llama-finetune)
add_executable(${TARGET} finetune.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET llama-finetune-lora)

if (NOT ANDROID)
    add_executable(${TARGET} finetune-lora.cpp)
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
else()
    # Build as Android native activity
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -u ANativeActivity_onCreate")
    add_library(${TARGET} SHARED finetune-lora.cpp)
    target_include_directories(${TARGET} PRIVATE ${ANDROID_NDK}/sources/android/native_app_glue)
    target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT} android native_app_glue log)
endif()

install(TARGETS ${TARGET} RUNTIME)
target_compile_features(${TARGET} PRIVATE cxx_std_11)