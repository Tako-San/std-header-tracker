cmake_minimum_required(VERSION 3.20)
project(std_header_tracker)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(LLVM CONFIG REQUIRED)
find_package(Clang CONFIG REQUIRED)
find_package(nlohmann_json REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
include(AddLLVM)
include(HandleLLVMOptions)

set(ic_llvm_dir ${LLVM_DIR})
set(ic_include_dirs
  ${LLVM_INCLUDE_DIRS}
  ${CLANG_INCLUDE_DIRS}
)

message(STATUS
  "IC: configuring for LLVM ${LLVM_VERSION} from ${ic_llvm_dir}")

option(IC_LINK_CLANG_DYLIB
  "Link against the clang dynamic library"
  ${CLANG_LINK_CLANG_DYLIB}
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(LLVM_LINK_COMPONENTS
  Option
  Support
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsInfos
)

add_llvm_executable(std-header-tracker src/main.cpp)

if(TARGET clang-resource-headers)
  add_dependencies(std-header-tracker clang-resource-headers)
endif()

set_target_properties(std-header-tracker PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_STANDARD 20
  CXX_EXTENSIONS OFF
)

separate_arguments(LLVM_DEFINITIONS_LIST
  NATIVE_COMMAND ${LLVM_DEFINITIONS}
)

target_compile_definitions(std-header-tracker
  PRIVATE ${LLVM_DEFINITIONS_LIST}
)

target_include_directories(std-header-tracker
  PRIVATE ${ic_include_dirs}
)

if(IC_LINK_CLANG_DYLIB)
  target_link_libraries(std-header-tracker
    PRIVATE clang-cpp
  )
else()
  target_link_libraries(std-header-tracker
    PRIVATE
    clangBasic
    clangLex
    clangAST
    clangSema
    clangFrontend
    clangFrontendTool
    clangDriver
    clangSerialization
  )
endif()

target_link_libraries(std-header-tracker
  PRIVATE nlohmann_json::nlohmann_json
)
