cmake_minimum_required(VERSION 3.0...3.26)


# Find and include Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# set(OpenBLAS_INCLUDE_DIRS "/usr/include/openblas")
# set(OpenBLAS_LIBRARIES "/usr/lib/x86_64-linux-gnu/libopenblas.so")
# include_directories(${OpenBLAS_INCLUDE_DIRS})
# find_package(OpenBLAS REQUIRED)

set(OpenBLAS_LIBRARIES "/usr/lib/x86_64-linux-gnu/libopenblas.so")
# include_directories(${OpenBLAS_INCLUDE_DIRS})

project(hnswlib
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include_directories(${CMAKE_SOURCE_DIR}/cnpy)
include_directories(${CMAKE_SOURCE_DIR}/otlib)

add_library(cnpy STATIC ${CMAKE_SOURCE_DIR}/cnpy/cnpy.cpp)
file(GLOB OT_LIB_SRC ${CMAKE_SOURCE_DIR}/otlib/*.cpp) 
add_library(otlib STATIC ${OT_LIB_SRC})
add_library(hnswlib INTERFACE)
add_library(hnswlib::hnswlib ALIAS hnswlib)

find_package(ZLIB REQUIRED)
include_directories(${ZLIB_INCLUDE_DIRS})

target_include_directories(hnswlib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Install
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/hnswlib
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(TARGETS hnswlib
    EXPORT hnswlibTargets)

install(EXPORT hnswlibTargets
    FILE hnswlibConfig.cmake
    NAMESPACE hnswlib::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hnswlib)

# Examples and tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(HNSWLIB_EXAMPLES "Build examples and tests." ON)
else()
    option(HNSWLIB_EXAMPLES "Build examples and tests." OFF)
endif()
if(HNSWLIB_EXAMPLES)
    set(CMAKE_CXX_STANDARD 11)

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      SET( CMAKE_CXX_FLAGS  "-Ofast -std=c++11 -DHAVE_CXX0X -openmp -lopenblas  -march=native -fpic -ftree-vectorize" )
      check_cxx_compiler_flag("-march=native" COMPILER_SUPPORT_NATIVE_FLAG)
      if(COMPILER_SUPPORT_NATIVE_FLAG)
        SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopenblas -march=native" )
        message("set -march=native flag")
      else()
        check_cxx_compiler_flag("-mcpu=apple-m1" COMPILER_SUPPORT_M1_FLAG)
        if(COMPILER_SUPPORT_M1_FLAG)
          SET( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mcpu=apple-m1 -lopenblas" )
          message("set -mcpu=apple-m1 flag")
        endif()
      endif()
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -lopenblas  -march=native -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0" )
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
      SET( CMAKE_CXX_FLAGS  "/O3 -DHAVE_CXX0X /W1 /openmp /EHsc" )
    endif()

    # examples
    add_executable(example_vecset_search examples examples/cpp/example_vecset_search.cpp)
    add_executable(example_vecset_search_gem examples examples/cpp/example_vecset_search_gem.cpp)
    
    target_link_libraries(example_vecset_search PRIVATE cnpy ${ZLIB_LIBRARIES})
    target_link_libraries(example_vecset_search_gem PRIVATE cnpy ${ZLIB_LIBRARIES})

    target_link_libraries(example_vecset_search PRIVATE otlib ${ZLIB_LIBRARIES}) 
    target_link_libraries(example_vecset_search_gem PRIVATE otlib ${ZLIB_LIBRARIES})  

    target_link_libraries(example_vecset_search PRIVATE ${OpenBLAS_LIBRARIES})
    target_link_libraries(example_vecset_search_gem PRIVATE ${OpenBLAS_LIBRARIES})

    # Link Eigen3
    target_link_libraries(example_vecset_search PRIVATE Eigen3::Eigen)
    target_link_libraries(example_vecset_search_gem PRIVATE Eigen3::Eigen)
    # add_executable(example_search examples/cpp/example_search.cpp)
    # target_link_libraries(example_search hnswlib)

    # add_executable(example_epsilon_search examples/cpp/example_epsilon_search.cpp)
    # target_link_libraries(example_epsilon_search hnswlib)

    # add_executable(example_multivector_search examples/cpp/example_multivector_search.cpp)
    # target_link_libraries(example_multivector_search hnswlib)

    # add_executable(example_filter examples/cpp/example_filter.cpp)
    # target_link_libraries(example_filter hnswlib)

    # add_executable(example_replace_deleted examples/cpp/example_replace_deleted.cpp)
    # target_link_libraries(example_replace_deleted hnswlib)

    # add_executable(example_mt_search examples/cpp/example_mt_search.cpp)
    # target_link_libraries(example_mt_search hnswlib)

    # add_executable(example_mt_filter examples/cpp/example_mt_filter.cpp)
    # target_link_libraries(example_mt_filter hnswlib)

    # add_executable(example_mt_replace_deleted examples/cpp/example_mt_replace_deleted.cpp)
    # target_link_libraries(example_mt_replace_deleted hnswlib)

    # # tests
    # add_executable(multivector_search_test tests/cpp/multivector_search_test.cpp)
    # target_link_libraries(multivector_search_test hnswlib)

    # add_executable(epsilon_search_test tests/cpp/epsilon_search_test.cpp)
    # target_link_libraries(epsilon_search_test hnswlib)

    # add_executable(test_updates tests/cpp/updates_test.cpp)
    # target_link_libraries(test_updates hnswlib)

    # add_executable(searchKnnCloserFirst_test tests/cpp/searchKnnCloserFirst_test.cpp)
    # target_link_libraries(searchKnnCloserFirst_test hnswlib)

    # add_executable(searchKnnWithFilter_test tests/cpp/searchKnnWithFilter_test.cpp)
    # target_link_libraries(searchKnnWithFilter_test hnswlib)

    # add_executable(multiThreadLoad_test tests/cpp/multiThreadLoad_test.cpp)
    # target_link_libraries(multiThreadLoad_test hnswlib)

    # add_executable(multiThread_replace_test tests/cpp/multiThread_replace_test.cpp)
    # target_link_libraries(multiThread_replace_test hnswlib)

    # add_executable(main tests/cpp/main.cpp tests/cpp/sift_1b.cpp)
    # target_link_libraries(main hnswlib)
endif()
