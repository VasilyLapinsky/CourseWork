cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME CourseWork)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})

if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/v0.16.1/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake"
                EXPECTED_HASH SHA256=396e16d0f5eabdc6a14afddbcfff62a54a7ee75c6da23f32f7a31bc85db23484
                TLS_VERIFY ON)
endif()

include("${CMAKE_BINARY_DIR}/conan.cmake")

conan_cmake_run(REQUIRES opencv/4.5.2 jsoncpp/1.9.4
                BASIC_SETUP CMAKE_TARGETS
                BUILD missing
                GENERATORS cmake_find_package_multi
                CONFIGURATION_TYPES "Debug;Release")

file(GLOB_RECURSE SOURCES  src/*.cpp src/*.cu)
file(GLOB_RECURSE HEADERS src/*.h)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "35;50;60;61;62;70;72;75")

cuda_add_executable(${PROJECT_NAME}
    ${SOURCES}
    ${HEADERS}
)

set_property(TARGET ${PROJECT_NAME} 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME}
        CONAN_PKG::opencv
        CONAN_PKG::jsoncpp
        ${CUDA_LIBRARIES}
        curand.lib
)