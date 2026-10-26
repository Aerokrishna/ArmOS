cmake_minimum_required(VERSION 3.8)
project(franka_custom_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Control UI (PyQt5) python package + executable
ament_python_install_package(franka_control_ui PACKAGE_DIR ui/franka_control_ui)
install(
  PROGRAMS scripts/franka_control_ui
  DESTINATION lib/${PROJECT_NAME}
)

# Admittance test publisher (the admittance node itself is armos_control's admittance_controller)
install(
  PROGRAMS bimanual/admittance_test_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Insertion (spiral search on top of the admittance node)
install(
  PROGRAMS insertion/spiral_search.py insertion/spiral_insertion_demo.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
