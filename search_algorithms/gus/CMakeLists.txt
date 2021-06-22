cmake_minimum_required(VERSION 2.8.11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE RELEASE)

project(GUS)

include_directories(.)

# Enable ExternalProject CMake module
include(ExternalProject)

# Include uint128_t - note; uint256_t can be included in a similar way if desired
ExternalProject_Add(
    uint128_t
    GIT_REPOSITORY https://github.com/calccrypto/uint128_t.git
    GIT_TAG master
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/uint128_t
    BUILD_COMMAND g++ -c -std=c++11 ../uint128_t/uint128_t.cpp -I ../uint128_t/ -O3 -o uint128t.o && ar rcs libuint128t.a uint128t.o
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    TEST_COMMAND ""
    UPDATE_DISCONNECTED 1
)
ExternalProject_Get_Property(uint128_t BINARY_DIR)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/uint128_t/src/)

add_library(libuint128 IMPORTED STATIC GLOBAL)
add_dependencies(libuint128 uint128_t)
set_target_properties(libuint128 PROPERTIES
    "IMPORTED_LOCATION" "${BINARY_DIR}/libuint128t.a"
)

# Include Eigen
ExternalProject_Add(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/eigen3
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    CONFIGURE_COMMAND ""
    TEST_COMMAND ""
)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/eigen3/src/eigen)

# Include gtest
ExternalProject_Add(
    gtest
    URL https://github.com/google/googletest/archive/master.zip
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(gtest SOURCE_DIR BINARY_DIR)

# Create a libgtest target to be used as a dependency by test programs
add_library(libgtest IMPORTED STATIC GLOBAL)
add_dependencies(libgtest gtest)
set_target_properties(libgtest PROPERTIES
    "IMPORTED_LOCATION" "${BINARY_DIR}/lib/libgtest.a"
)

# Create a libgmock target to be used as a dependency by test programs
add_library(libgmock IMPORTED STATIC GLOBAL)
add_dependencies(libgmock gtest)
set_target_properties(libgmock PROPERTIES
    "IMPORTED_LOCATION" "${BINARY_DIR}/lib/libgmock.a"
)

# Build gtest
include_directories("${SOURCE_DIR}/googletest/include"
                    "${SOURCE_DIR}/googlemock/include")

add_executable(selector_test gus/selector_test.cc)
add_dependencies(selector_test eigen)
target_link_libraries(selector_test libgmock libgtest pthread)
target_link_libraries(selector_test libuint128)
