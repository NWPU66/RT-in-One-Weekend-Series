# subdirs(${CMAKE_CURRENT_SOURCE_DIR}/util)

# libraries
# find_package(glm CONFIG REQUIRED)

# cuda compiler
enable_language(CUDA)
# set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
# file(GLOB_RECURSE CUDA_LIBS "E:/cuda118_x64_lib/*.lib")
# message(${CUDA_LIBS})
# link_directories(${CUDA_ROOT}/lib/x64)

# set(NVCC_FLAGS -g -G)

file(GLOB UTIL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

file(GLOB_RECURSE CH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Ch*)
foreach(CH_FILE ${CH_FILES})
    #获取文件名
    get_filename_component(CH_FILE_NAME ${CH_FILE} NAME_WE)
    #生成可执行文件
    add_executable(${CH_FILE_NAME} 
        ${CH_FILE})
    #添加头文件目录
    target_include_directories(${CH_FILE_NAME} PRIVATE
        ${VCPKG_INCLUDE_DIR}
        ${SRC_DIR}
        # ${CUDA_ROOT}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
    #添加链接库
    target_link_libraries(${CH_FILE_NAME} PRIVATE 
        # glm::glm
        # ${CUDA_LIBS}
        )
    #安装
    install(TARGETS ${CH_FILE_NAME} RUNTIME DESTINATION ./)
endforeach()