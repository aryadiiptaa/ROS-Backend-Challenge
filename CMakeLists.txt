cmake_minimum_required(VERSION 3.8)

project(robot_ros_backend)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
  )
endif()

find_package(ament_cmake REQUIRED)

# Memasang folder resource package ke install directory
install(
  DIRECTORY
    launch
    urdf
    worlds
    config
    maps
    behavior_trees
  DESTINATION
    share/${PROJECT_NAME}
)

install(
  PROGRAMS
    maps/coverage_navigator.py
  DESTINATION
    lib/${PROJECT_NAME}
)

# Install modular Python helpers beside the executable so both source and
# installed execution use the same imports.
install(
  DIRECTORY
    maps/coverage_modules
  DESTINATION
    lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)

  set(
    ament_cmake_copyright_FOUND TRUE
  )

  set(
    ament_cmake_cpplint_FOUND TRUE
  )

  ament_lint_auto_find_test_dependencies()
endif()

ament_package()