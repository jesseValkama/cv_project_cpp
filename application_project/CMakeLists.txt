add_executable (application_project) 
target_sources (application_project 
                PRIVATE
                        "application_project.cpp"
                        "src/augmentations/geometry.cpp"
                        "src/database/database.cpp" "src/database/database_map.cpp"
                        "src/database/param_info.cpp" "src/database/query_template_manager.cpp"
                        "src/database/database_helpers.cpp"
                        "src/datasets/loader_funcs.cpp" "src/datasets/mnist.cpp"
                        "src/datasets/cifar10.cpp" "src/datasets/data_helper.cpp"
                        "src/settings.cpp"
                        "src/models/common.cpp" "src/models/model_wrapper.cpp"
                        "src/models/lenet.cpp" "src/models/resnet.cpp"
                        "src/metrics/classification_metrics.cpp" "src/metrics/timer.cpp"
                        "src/functions/common.cpp" "src/functions/experiments_database.cpp"
                        "src/functions/train_lenet.cpp"
                        "src/functions/inference.cpp" "src/functions/handle_args.cpp"
                        "src/functions/gradcam.cpp" "src/functions/visualise.cpp"
                        "src/optims/scheds/warmup.cpp"
                        "src/plots/plots.cpp"
)
if (LINUX)
    include(cmake/linux/linux.cmake)
elseif (WIN32)
    include(cmake/windows/windows.cmake)
endif (LINUX)
set_property(TARGET application_project PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#add_subdirectory(tests)