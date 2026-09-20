cmake_minimum_required(VERSION 3.18)
project(gem VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_path(CBLAS_INCLUDE_DIR cblas.h REQUIRED)
find_library(OPENBLAS_LIBRARY NAMES openblas REQUIRED)
option(GEM_NATIVE_ARCH "Compile for the current CPU" OFF)

add_library(gem_ot STATIC src/distance/otlib/EMD_wrapper.cpp)
target_include_directories(gem_ot PUBLIC src/distance/otlib)
add_library(gem_cnpy STATIC src/io/cnpy/cnpy.cpp)
target_include_directories(gem_cnpy PUBLIC src/io/cnpy)
target_link_libraries(gem_cnpy PUBLIC ZLIB::ZLIB)

add_library(gem STATIC
    src/index.cpp
    src/encoding/corpus.cpp src/encoding/assignment.cpp
    src/build/cluster_graph.cpp src/build/finalize.cpp
    src/search/routing.cpp src/search/traversal.cpp src/search/rerank.cpp src/search/workspace.cpp
    src/distance/dot_table.cpp src/distance/maxsim.cpp src/distance/qemd.cpp
    src/graph/index.cpp
    src/io/numpy.cpp src/io/graph_index.cpp)
add_library(gem::gem ALIAS gem)
target_compile_features(gem PUBLIC cxx_std_17)
target_include_directories(gem PUBLIC include PRIVATE src ${CBLAS_INCLUDE_DIR})
target_link_libraries(gem PRIVATE gem_ot gem_cnpy Eigen3::Eigen OpenMP::OpenMP_CXX
                                 Threads::Threads ${OPENBLAS_LIBRARY})
target_compile_definitions(gem PRIVATE EIGEN_DONT_PARALLELIZE)
if(GEM_NATIVE_ARCH)
    target_compile_options(gem PRIVATE -march=native)
endif()

add_library(gem_app_config STATIC apps/config.cpp)
target_compile_features(gem_app_config PUBLIC cxx_std_17)
target_include_directories(gem_app_config PUBLIC apps include)
add_executable(gem_run apps/gem_run.cpp apps/benchmark.cpp)
target_include_directories(gem_run PRIVATE ${CBLAS_INCLUDE_DIR})
target_link_libraries(gem_run PRIVATE gem::gem gem_app_config OpenMP::OpenMP_CXX ${OPENBLAS_LIBRARY})

if(BUILD_TESTING)
    add_executable(gem_io_config_tests tests/io_config_test.cpp)
    target_link_libraries(gem_io_config_tests PRIVATE gem::gem gem_app_config gem_cnpy)
    add_test(NAME gem_io_config COMMAND gem_io_config_tests)
    add_executable(gem_upstream_reference tests/upstream_reference.cpp)
    target_include_directories(gem_upstream_reference PRIVATE src ${CBLAS_INCLUDE_DIR})
    target_link_libraries(gem_upstream_reference PRIVATE gem::gem gem_app_config gem_ot gem_cnpy
                         Eigen3::Eigen OpenMP::OpenMP_CXX ${OPENBLAS_LIBRARY})
    if(GEM_NATIVE_ARCH)
        target_compile_options(gem_upstream_reference PRIVATE -march=native)
    endif()
    add_executable(gem_tests tests/library_test.cpp)
    target_include_directories(gem_tests PRIVATE src)
    target_link_libraries(gem_tests PRIVATE gem::gem Eigen3::Eigen Threads::Threads)
    add_test(NAME gem_library COMMAND gem_tests)
    set_tests_properties(gem_library PROPERTIES ENVIRONMENT "OPENBLAS_NUM_THREADS=1;OMP_NUM_THREADS=1")
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
            RESULT_VARIABLE GEM_NUMPY_MISSING OUTPUT_QUIET ERROR_QUIET)
        if(NOT GEM_NUMPY_MISSING)
            add_test(NAME gem_preprocess COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/preprocess_test.py")
        endif()
    endif()
endif()

option(GEM_BUILD_LEGACY_EXAMPLE "Build the example used by the original README" OFF)
if(GEM_BUILD_LEGACY_EXAMPLE)
    add_executable(example_vecset_search_gem tests/reference/hnswlib/examples/cpp/example_vecset_search_gem.cpp)
    target_include_directories(example_vecset_search_gem PRIVATE src ${CBLAS_INCLUDE_DIR})
    target_link_libraries(example_vecset_search_gem PRIVATE gem_ot gem_cnpy Eigen3::Eigen
        OpenMP::OpenMP_CXX ${OPENBLAS_LIBRARY})
    target_compile_features(example_vecset_search_gem PRIVATE cxx_std_17)
    set_target_properties(example_vecset_search_gem PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
    if(GEM_NATIVE_ARCH)
        target_compile_options(example_vecset_search_gem PRIVATE -march=native)
    endif()
endif()
