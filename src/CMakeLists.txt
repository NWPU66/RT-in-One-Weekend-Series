set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# add sub directories
subdirs(
    ${CMAKE_CURRENT_SOURCE_DIR}/RT_in_One_Weekend
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDA_RT_in_One_Weekend
    ${CMAKE_CURRENT_SOURCE_DIR}/CUDA_RT_The_Next_Week
)

# helloworld programe
# find_package(glm CONFIG REQUIRED)
# find_package(imgui CONFIG REQUIRED)
add_executable(helloworld helloworld.cpp)
target_include_directories(helloworld PRIVATE
    ${VCPKG_INCLUDE_DIR}
    ${SRC_DIR})
target_link_libraries(helloworld PRIVATE 
    # imgui::imgui
    #glm::glm
    )
install(TARGETS helloworld RUNTIME DESTINATION ./)
