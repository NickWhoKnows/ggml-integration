cmake_minimum_required(VERSION 3.20)
project(ggml_integration)
set(CMAKE_CXX_STANDARD 20)
# Enable Metal on macOS (you have an M4 Mac)
set(GGML_METAL ON)
add_subdirectory(include/ggml)
add_executable(app src/main.cpp
    src/gguf.h
    src/gguf.cpp)
target_link_libraries(app PRIVATE ggml)
target_include_directories(app PRIVATE include/ggml/include)