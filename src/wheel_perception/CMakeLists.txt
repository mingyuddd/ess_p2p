cmake_minimum_required(VERSION 3.10)
project(wheel_perception)

# --- 1. 基础配置 ---
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# --- 2. 寻找依赖 ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(rclcpp_components REQUIRED) # [必须] 组件化依赖
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(wheel_msgs REQUIRED)        # [必须] 自定义消息包
find_package(OpenCV REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# --- 3. CUDA & TensorRT & ZED 配置 ---
if(NOT CMAKE_CUDA_COMPILER)
  if(DEFINED ENV{CUDACXX})
    set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}" CACHE FILEPATH "CUDA compiler" FORCE)
  else()
    find_program(WHEEL_NVCC
      NAMES nvcc
      HINTS
        /usr/local/cuda/bin
        /usr/local/cuda-12.6/bin
        /usr/local/cuda-12.5/bin
        /usr/local/cuda-12.4/bin
        /usr/local/cuda-12.3/bin
        /usr/local/cuda-12.2/bin
        /usr/local/cuda-12.1/bin
        /usr/local/cuda-12.0/bin
        /usr/local/cuda-11.8/bin
    )
    if(WHEEL_NVCC)
      set(CMAKE_CUDA_COMPILER "${WHEEL_NVCC}" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
  endif()
endif()

enable_language(CUDA)
find_package(CUDA REQUIRED)
set(CUDA_LIB_PATH "/usr/local/cuda-11.8/lib64") # 根据实际路径调整

# 避免使用 `native`：在部分 CMake + CUDA 11.8 组合下会被翻译成
# nvcc 不识别的 `compute_native`，从而导致编译失败。
# 如有需要，可在 colcon/cmake 命令行中覆盖，例如：
#   --cmake-args -DWHEEL_CUDA_ARCHITECTURES=86
set(WHEEL_CUDA_ARCHITECTURES "" CACHE STRING
  "Semicolon-separated CUDA architectures for wheel_perception (e.g. 72;75;86;87)")

if(NOT WHEEL_CUDA_ARCHITECTURES)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|ARM64)$")
    set(WHEEL_CUDA_ARCHITECTURES "72;87")
  else()
    set(WHEEL_CUDA_ARCHITECTURES "75;86")
  endif()
endif()

message(STATUS "wheel_perception CUDA architectures: ${WHEEL_CUDA_ARCHITECTURES}")

find_library(CUDART_LIB NAMES cudart HINTS ${CUDA_LIB_PATH})
find_library(CUDA_LIB NAMES cuda HINTS ${CUDA_LIB_PATH} /usr/lib/wsl/lib)

# TensorRT
set(TRT_ROOT "/usr/local/cuda-12.2/targets/x86_64-linux")  # TensorRT 10.9（CUDA 12 运行库） # 请确认路径正确
find_library(NVINFER_LIB nvinfer PATHS ${TRT_ROOT}/lib NO_DEFAULT_PATH)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS ${TRT_ROOT}/lib NO_DEFAULT_PATH)

# jsoncpp（usb_stereo_driver.cpp 解析校准文件用，缺失会导致运行时 symbol lookup error）
find_library(JSONCPP_LIB NAMES jsoncpp)

# ZED SDK
set(ZED_INCLUDE_DIRS "/usr/local/zed/include")
set(ZED_LIBRARY_DIR "/usr/local/zed/lib")
find_library(ZED_LIB NAMES sl_zed HINTS ${ZED_LIBRARY_DIR})

include_directories(
  ${TRT_ROOT}/include   # TensorRT 10 头文件（必须最前！cuda-11.8 targets 里还有 TRT 8.6 的 NvInfer.h，顺序不对会编出 TRT 8 宏）
  include
  ${CUDA_INCLUDE_DIRS}
  ${ZED_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# =========================================================
# 4. 核心算法库 (Wheel Core)
# =========================================================
add_library(wheel_core SHARED
  src/core/zed/zed_driver.cpp
  src/core/usb/usb_stereo_driver.cpp
  src/core/usb/ess_depth_engine.cpp
  src/core/ai_engine.cpp
  src/core/obstacle_fusion.cu
  src/core/mask_postprocess.cu
  src/core/preprocess.cu
  src/core/usb/ess_preprocess.cu
  src/core/usb/depth_to_cloud.cu
  src/core/global_flags.cpp  # 新增
)

if(CMAKE_CUDA_COMPILER)
  set_property(TARGET wheel_core PROPERTY CUDA_ARCHITECTURES ${WHEEL_CUDA_ARCHITECTURES})
endif()

# 链接核心库依赖
target_link_libraries(wheel_core
  ${ZED_LIB}
  ${CUDA_LIB}
  ${CUDART_LIB}
  ${NVINFER_LIB}
  ${NVINFER_PLUGIN_LIB}
  ${JSONCPP_LIB}
  ${OpenCV_LIBS}
)

# [关键] 核心库也需要 ROS 依赖
ament_target_dependencies(wheel_core
  rclcpp
  rclcpp_lifecycle
  sensor_msgs
  cv_bridge
)

# =========================================================
# 5. 感知节点组件 (FusionNode) - 编译为 SHARED 库
# =========================================================
# 注意：这里绝不能用 add_executable，必须是 add_library
add_library(fusion_node SHARED src/fusion_node.cpp)

target_include_directories(fusion_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

find_package(Eigen3 REQUIRED)

# 在包含目录里添加 Eigen 的路径
include_directories(
  include
  ${Eigen3_INCLUDE_DIRS}
)

# 链接核心库
target_link_libraries(fusion_node wheel_core )

# 添加 ROS 依赖
ament_target_dependencies(fusion_node
  rclcpp
  rclcpp_lifecycle
  rclcpp_components  # 组件化核心
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  wheel_msgs         # 你的自定义消息
  nav_msgs           # 导航消息
)

# [关键] 注册为组件
rclcpp_components_register_nodes(fusion_node "FusionNode")

# =========================================================
# 6. 控制节点组件 (ControllerNode)
# =========================================================
add_library(controller_node SHARED src/controller_node.cpp)


target_include_directories(controller_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(controller_node wheel_core)
ament_target_dependencies(controller_node geometry_msgs nav_msgs tf2 tf2_geometry_msgs
  rclcpp
  rclcpp_components
  wheel_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_geometry_msgs
)

# [关键] 注册为组件
# 注意：代码里是 wheel_control::ControllerNode，这里名字要对应
rclcpp_components_register_nodes(controller_node "wheel_control::ControllerNode")

# =========================================================
# 6.5 远程 ESS 服务端节点组件 (RemoteEssServer) - 双机模式
# 目标机部署：订阅 ess_remote/stereo_input(JPEG 拼接图) ->
# 推理 -> 发布 ess_remote/depth(16UC1 毫米)
# =========================================================
add_library(remote_ess_server SHARED src/remote_ess_server.cpp)

target_include_directories(remote_ess_server PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(remote_ess_server wheel_core ${OpenCV_LIBS})
ament_target_dependencies(remote_ess_server
  rclcpp
  rclcpp_components
  sensor_msgs
  std_msgs
  cv_bridge
  wheel_msgs
)

# 注册为组件（launch/remote_ess_server.launch.py 加载）
rclcpp_components_register_nodes(remote_ess_server "RemoteEssServer")

# =========================================================
# 7. 安装规则
# =========================================================
install(TARGETS
  wheel_core
  fusion_node
  controller_node
  remote_ess_server
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch camera ess
  DESTINATION share/${PROJECT_NAME}
)

# 安装 Launch 文件
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/sub.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install config directory
install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

# Install launch directory (顺便检查一下 launch 也没漏)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
