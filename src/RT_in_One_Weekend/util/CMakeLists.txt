# util lib
file(GLOB_RECURSE SRC ${CMAKE_CURRENT_SOURCE_DIR}/*)
add_library(util STATIC ${SRC})
target_include_directories(util PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${VCPKG_INCLUDE_DIR})
target_link_libraries(util PRIVATE 
    glm::glm)