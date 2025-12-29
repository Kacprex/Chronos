cmake_minimum_required(VERSION 3.20)
project(chronos_engine LANGUAGES CXX)

option(CHRONOS_WITH_ONNX "Enable ONNX Runtime inference" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (MSVC)
  add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()
add_executable(chronos_engine
    src/main.cpp
    src/uci.cpp

    src/board/board.cpp
    src/board/move.cpp
    src/board/movegen.cpp
    src/board/attacks.cpp

    src/search/search.cpp
    src/search/tt.cpp

    src/eval/eval.cpp

    src/util/zobrist.cpp
    src/util/chrono_time.cpp
    src/util/paths.cpp
    src/util/logging.cpp

    src/nn/nn_interface.cpp
    src/nn/move_index.cpp
src/nn/onnx_runtime_impl.cpp
)

target_include_directories(chronos_engine PRIVATE include)


if(CHRONOS_WITH_ONNX)
  target_compile_definitions(chronos_engine PRIVATE CHRONOS_WITH_ONNX=1)

  # Phase 6: Optional ONNX Runtime integration.
  # On Windows, download ONNX Runtime (CPU or GPU) and set CHRONOS_ONNX_ROOT
  # to the extracted folder that contains include/ and lib/ (and bin/ for the DLL).
  set(CHRONOS_ONNX_ROOT "" CACHE PATH "Path to ONNX Runtime root folder")
  if(CHRONOS_ONNX_ROOT)
    target_include_directories(chronos_engine PRIVATE "${CHRONOS_ONNX_ROOT}/include")
    target_link_directories(chronos_engine PRIVATE "${CHRONOS_ONNX_ROOT}/lib")
    target_link_libraries(chronos_engine PRIVATE onnxruntime)
  else()
    message(WARNING "CHRONOS_WITH_ONNX=ON but CHRONOS_ONNX_ROOT is empty. Set -DCHRONOS_ONNX_ROOT=... to link.")
  endif()
endif()

if(MSVC)
  target_compile_options(chronos_engine PRIVATE /O2 /W4)
else()
  target_compile_options(chronos_engine PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
