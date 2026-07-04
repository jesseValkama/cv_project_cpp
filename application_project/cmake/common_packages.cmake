include(${CMAKE_CURRENT_LIST_DIR}/fetch_content.cmake)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_library(sqlite3 
    STATIC ${SQLITE3_ROOT}/sqlite3.c
)
target_include_directories(sqlite3
    PUBLIC ${SQLITE3_ROOT}
)

target_include_directories(application_project 
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OPENCV_INCLUDE_DIRS}       
)
target_link_libraries(application_project 
    PRIVATE "${TORCH_LIBRARIES}" 
    PRIVATE ${OpenCV_LIBS} 
    PUBLIC yaml-cpp::yaml-cpp
    PRIVATE sqlite3
)