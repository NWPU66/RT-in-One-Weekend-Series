subdirs(${CMAKE_CURRENT_SOURCE_DIR}/util)

# libraries
find_package(glm CONFIG REQUIRED)

file(GLOB_RECURSE CH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Ch*.cpp)
foreach(CH_FILE ${CH_FILES})
    #获取文件名
    get_filename_component(CH_FILE_NAME ${CH_FILE} NAME_WE)
    #生成可执行文件
    add_executable(${CH_FILE_NAME} 
        ${CH_FILE})
    #添加头文件目录
    target_include_directories(${CH_FILE_NAME} PRIVATE
        ${VCPKG_INCLUDE_DIR}
        ${SRC_DIR})
    #添加链接库
    target_link_libraries(${CH_FILE_NAME} PRIVATE 
        glm::glm
        util)
    #安装
    install(TARGETS ${CH_FILE_NAME} RUNTIME DESTINATION ./)
endforeach()