cmake_minimum_required(VERSION 3.12)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
	message("current platform: Linux ")
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
	message("current platform: Windows ")
    set(CMAKE_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(VCPKG_INCLUDE_DIR "E:/vcpkg/installed/x64-windows/include")
else()
	message("current platform: unkonw ")
endif()

set(CMAKE_CXX_STANDARD 17)
project(RT-in-One-Weekend-Series
    VERSION 0.1.0
    DESCRIPTION "RT-in-One-Weekend-Series"
    HOMEPAGE_URL "https://github.com/NWPU66/RT-in-One-Weekend-Series"
    LANGUAGES CXX CUDA)

# add subdirectory
subdirs(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
subdirs(${CMAKE_CURRENT_SOURCE_DIR}/src)

# install path
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)

# cmake cli
# cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -G "Ninja" -DCMAKE_MAKE_PROGRAM=/data/syj/yzj/ninja