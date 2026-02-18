add_executable (application_project "application_project.cpp"
                                    "src/augmentations/geometry.cpp"
                                    "src/datasets/loader_funcs.cpp" "src/datasets/mnist.cpp"
                                    "src/datasets/cifar10.cpp" "src/datasets/data_helper.cpp"
                                    "src/settings.cpp"
                                    "src/models/common.cpp" "src/models/model_wrapper.cpp"
                                    "src/models/lenet.cpp" "src/models/resnet.cpp"
                                    "src/metrics/classification_metrics.cpp"
                                    "src/functions/common.cpp" "src/functions/database.cpp"
                                    "src/functions/train_lenet.cpp"
                                    "src/functions/inference.cpp" "src/functions/handle_args.cpp"
                                    "src/functions/gradcam.cpp" "src/functions/visualise.cpp"
                                    "src/optims/scheds/warmup.cpp"
                                    "src/plots/plots.cpp"
)

include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master
)
FetchContent_MakeAvailable(yaml-cpp)

set(CMAKE_PREFIX_PATH "D:/cpptools/libtorch;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6;D:/cpptools/opencv/build")
set(SQLITE3_ROOT "D:/cpptools/sqlite3")

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

set_property(TARGET application_project PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# uses shared library optimisations, you might need to add additional dlls here, or change torch to ${TORCH_DLLS} (slower)
if (MSVC)
    set(TORCH_DLLS
        "${TORCH_INSTALL_PREFIX}/lib/torch_cuda.dll"
        "${TORCH_INSTALL_PREFIX}/lib/torch_cpu.dll"
        "${TORCH_INSTALL_PREFIX}/lib/torch.dll"
        "${TORCH_INSTALL_PREFIX}/lib/c10.dll"
        "${TORCH_INSTALL_PREFIX}/lib/c10_cuda.dll"
    )
    set(OPENCV_DLLS) # makes loading faster
    add_custom_command(TARGET application_project POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:application_project>
    )
endif(MSVC)

#add_subdirectory(tests)