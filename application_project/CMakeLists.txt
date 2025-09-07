add_executable (application_project "application_project.cpp" "application_project.h" 
                                    "datasets/loader_funcs.cpp" "datasets/mnist.cpp"
                                    "settings.cpp"
                                    "models/common.cpp" "models/lenet.cpp"
                                    "functions/common.cpp" "functions/train_lenet.cpp"
                                    "functions/inference.cpp" "functions/handle_args.cpp"
                                    "functions/gradcam.cpp" "functions/visualise.cpp"
)

set(CMAKE_PREFIX_PATH "D:/cpptools/libtorch;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6;D:/cpptools/opencv/build")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${OPENCV_INCLUDE_DIRS}
)
target_link_libraries(application_project "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

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