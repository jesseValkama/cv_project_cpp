# presuming that this project works at some point
# cuda 12.6, libtorch 2.8.0 debug (good luck getting that one), and opencv 4.7.0
# Add source to this project's executable.

add_executable (application_project "application_project.cpp" "application_project.h" 
                                    "datasets/loader_funcs.cpp"
                                    "settings.cpp"
                                    "models/common.cpp" "models/lenet.cpp"
                                    "datasets/mnist.cpp"
                                    "functions/common.cpp" 
                                    "functions/train_lenet.cpp")

set(CMAKE_PREFIX_PATH "D:/cpptools/libtorch;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6;D:/cpptools/opencv/build")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
target_link_libraries(application_project "${TORCH_LIBRARIES}" ${OpenCV_LIBS})

set_property(TARGET application_project PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# uses shared library optimisations, you might need to add additional dlls here, or change torch to ${TORCH_DLLS} (slower)
if (MSVC)
    set(TORCH_DLLS
        "${TORCH_INSTALL_PREFIX}/lib/torch_cuda_d.dll"
        "${TORCH_INSTALL_PREFIX}/lib/torch_cpu_d.dll"
        "${TORCH_INSTALL_PREFIX}/lib/torch_d.dll"
        "${TORCH_INSTALL_PREFIX}/lib/c10_d.dll"
        "${TORCH_INSTALL_PREFIX}/lib/c10_cuda_d.dll"
        "${TORCH_INSTALL_PREFIX}/lib/nvToolsExt64_1.dll"
    )

    set(OPENCV_DLLS
        "${OpenCV_INSTALL_PREFIX}/bin/opencv_core470d.dll"
        "${OpenCV_INSTALL_PREFIX}/bin/opencv_imgproc470d.dll"
        "${OpenCV_INSTALL_PREFIX}/bin/opencv_imgcodecs470d.dll"
    )

    add_custom_command(TARGET application_project POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_DLLS}
            ${OPENCV_DLLS}
            $<TARGET_FILE_DIR:application_project>
    )
endif(MSVC)

#legacy code that i am too scared to delete, used if needed to (again) reinstall all, doesn't use shared
# library optimisations, but atleast loads everything
#if (MSVC)
#    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll/")
#    add_custom_command(TARGET application_project
#    POST_BUILD
#		COMMAND ${CMAKE_COMMAND} -E copy_if_different
#				${TORCH_DLLS}
#		$<TARGET_FILE_DIR:application_project>
#    )
#endif(MSVC)
