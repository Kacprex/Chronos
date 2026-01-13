add_executable(chronos_encode_fen
    src/encode_fen_main.cpp
    src/fen.cpp
    src/encoding.cpp
)

target_include_directories(chronos_encode_fen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
