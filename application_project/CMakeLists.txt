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
                        "src/datasets/inference.cpp" "src/functions/experiments_database.cpp"
                        "src/models/common.cpp" "src/models/get_model.cpp"
                        "src/models/model_wrapper.cpp"
                        "src/models/lenet.cpp" "src/models/resnet.cpp"
                        "src/metrics/classification_metrics.cpp" "src/metrics/timer.cpp"
                        "src/inference/inference.cpp"
                        "src/optims/scheds/warmup.cpp" "src/plots/plots.cpp"
                        "src/settings/handle_args.cpp" "src/settings/settings.cpp"
                        "src/train/common.cpp" "src/train/early_stop.cpp"
                        "src/train/grad_scaler.cpp" "src/train/run_loop.cpp"
                        "src/train/test.cpp" "src/train/train.cpp"
                        "src/train/validate.cpp" "src/visualise/common.cpp"
                        "src/visualise/feature_map_visualiser.cpp" "src/visualise/gradcam.cpp"
)
if (LINUX)
    include(cmake/linux/linux.cmake)
elseif (WIN32)
    include(cmake/windows/windows.cmake)
endif ()
set_property(TARGET application_project PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#add_subdirectory(tests)