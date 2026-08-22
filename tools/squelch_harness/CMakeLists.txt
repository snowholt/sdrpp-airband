cmake_minimum_required(VERSION 3.13)
project(squelch_harness)

find_package(PkgConfig REQUIRED)
pkg_check_modules(VOLK REQUIRED volk)
find_package(Threads REQUIRED)

add_executable(squelch_unit_tests src/unit_tests.cpp)
target_compile_features(squelch_unit_tests PRIVATE cxx_std_17)
target_include_directories(squelch_unit_tests PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/core/src"
    ${VOLK_INCLUDE_DIRS})
target_link_directories(squelch_unit_tests PRIVATE ${VOLK_LIBRARY_DIRS})
target_link_libraries(squelch_unit_tests PRIVATE ${VOLK_LIBRARIES} Threads::Threads)

add_executable(squelch_sweep src/harness_main.cpp)
target_compile_features(squelch_sweep PRIVATE cxx_std_17)
target_include_directories(squelch_sweep PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_SOURCE_DIR}/core/src"
    ${VOLK_INCLUDE_DIRS})
target_link_directories(squelch_sweep PRIVATE ${VOLK_LIBRARY_DIRS})
target_link_libraries(squelch_sweep PRIVATE ${VOLK_LIBRARIES} Threads::Threads)
