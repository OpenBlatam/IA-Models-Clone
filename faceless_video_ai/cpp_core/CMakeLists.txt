cmake_minimum_required(VERSION 3.20)
project(faceless_video_cpp VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat
    libavcodec
    libavutil
    libswscale
    libswresample
    libavfilter
)
pkg_check_modules(OPENCV REQUIRED IMPORTED_TARGET opencv4)

# Optional: Find pybind11 for Python bindings
find_package(pybind11 CONFIG)

# Source files
set(SOURCES
    src/video_processor.cpp
    src/audio_processor.cpp
    src/effects_engine.cpp
    src/codec_wrapper.cpp
    src/transitions.cpp
    src/filters.cpp
)

set(HEADERS
    include/video_processor.hpp
    include/audio_processor.hpp
    include/effects_engine.hpp
    include/codec_wrapper.hpp
    include/transitions.hpp
    include/filters.hpp
    include/common.hpp
)

# Main library
add_library(faceless_video_cpp SHARED ${SOURCES})
target_include_directories(faceless_video_cpp PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(faceless_video_cpp PUBLIC
    PkgConfig::FFMPEG
    PkgConfig::OPENCV
)

# Python bindings (if pybind11 found)
if(pybind11_FOUND)
    pybind11_add_module(faceless_video_native bindings/python_bindings.cpp)
    target_link_libraries(faceless_video_native PRIVATE faceless_video_cpp)
endif()

# Install targets
install(TARGETS faceless_video_cpp
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${HEADERS} DESTINATION include/faceless_video)

# Testing
enable_testing()
add_subdirectory(tests EXCLUDE_FROM_ALL)




