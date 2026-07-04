include(${CMAKE_CURRENT_LIST_DIR}/../common_packages.cmake)
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