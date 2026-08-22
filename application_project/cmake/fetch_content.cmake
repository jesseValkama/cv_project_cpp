include(FetchContent)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG master
)
if (MSVC)
  set(YAML_MSVC_SHARED_RT ON)
endif ()
FetchContent_MakeAvailable(yaml-cpp)
