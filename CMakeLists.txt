# ==============================================================================
# CMake 配置文件
# 项目：IBVS_Teleoperation_UR - 基于图像的视觉伺服遥操作系统
# 配置逻辑：基础配置 → 依赖查找 → 包含目录设置 → 源文件设置 → 构建配置 → 链接配置
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. 基础配置
# ------------------------------------------------------------------------------

# 指定所需的 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言
project(IBVS_Teleoperation_UR CXX)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # 强制要求 C++17 标准
set(CMAKE_CXX_EXTENSIONS OFF)         # 禁用编译器特定的扩展，确保标准兼容性

# ------------------------------------------------------------------------------
# 2. 依赖查找
# ------------------------------------------------------------------------------

# ==============================================================================
# 2.1 VISP (Visual Servoing Platform) 依赖
# ==============================================================================

# 设置 VISP 库的安装路径
set(VISP_DIR "/home/kun/visp-ws/visp-build")

# 验证 VISP_DIR 是否存在
if(EXISTS ${VISP_DIR})
    message(STATUS "VISP_DIR 存在: ${VISP_DIR}")
else()
    message(WARNING "VISP_DIR 不存在: ${VISP_DIR}")
endif()

# 查找 VISP 库及其组件
# REQUIRED 表示必须找到，否则构建失败
find_package(VISP REQUIRED COMPONENTS 
    visp_core              # 核心功能库
    visp_gui               # 图形用户界面库
    visp_visual_features   # 视觉特征库
    visp_robot             # 机器人控制库
    visp_sensor            # 传感器接口库
    visp_detection         # 目标检测库
    visp_vs                # 视觉伺服库
)

# 打印 VISP 相关信息，用于调试和验证
message(STATUS "VISP 版本: ${VISP_VERSION}")
message(STATUS "VISP 库: ${VISP_LIBRARIES}")

# ==============================================================================
# 2.2 OpenCV 依赖
# ==============================================================================

# 查找 OpenCV 库（必需）
# OpenCV 用于图像处理和计算机视觉任务
find_package(OpenCV REQUIRED)

# ==============================================================================
# 2.3 CUDA 依赖 (可选)
# ==============================================================================

# 查找 CUDA（QUIET 表示静默查找，找不到不会报错）
# CUDA 用于 GPU 加速，主要用于 YOLO 检测
find_package(CUDA QUIET)

if(CUDA_FOUND)
    # 找到 CUDA，打印相关信息
    message(STATUS "CUDA 版本: ${CUDA_VERSION}")
    message(STATUS "CUDA 包含目录: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA 库: ${CUDA_LIBRARIES}")
else()
    # 未找到 CUDA，给出警告（不影响构建）
    message(WARNING "CUDA 未找到，YOLO 检测将不可用")
endif()

# ==============================================================================
# 2.4 TensorRT 依赖 (可选)
# ==============================================================================

# 设置 TensorRT 根目录
# 优先使用环境变量 TENSORRT_DIR，否则使用默认路径
if(DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_ROOT $ENV{TENSORRT_DIR})
    message(STATUS "找到 TensorRT 库: ${TENSORRT_ROOT}")
else()
    # 默认路径（兼容性设置）
    set(TENSORRT_ROOT "$ENV{HOME}/TensorRT-8.6")
    message(WARNING "未找到 TensorRT_DIR 环境变量，尝试默认位置: ${TENSORRT_ROOT}")
endif()

# 查找 TensorRT 库文件
# HINTS 指定搜索路径，提高查找效率
find_library(NVINFER_LIBRARY nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(NVPARSERS_LIBRARY nvparsers HINTS ${TENSORRT_ROOT}/lib)
find_library(NVONNXPARSERS_LIBRARY nvonnxparser HINTS ${TENSORRT_ROOT}/lib)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin HINTS ${TENSORRT_ROOT}/lib)

# 验证 TensorRT 是否找到
if(NVINFER_LIBRARY)
    # 找到 TensorRT，定义 HAVE_TENSORRT 宏
    # 这个宏用于条件编译，在没有 TensorRT 时提供替代实现
    message(STATUS "TensorRT 库: ${NVINFER_LIBRARY}")
    add_definitions(-DHAVE_TENSORRT)
else()
    # 未找到 TensorRT，给出警告（不影响构建）
    message(WARNING "TensorRT 未找到，YOLO 检测将不可用")
endif()

# ------------------------------------------------------------------------------
# 3. 包含目录设置
# ------------------------------------------------------------------------------

# 添加 TensorRT 包含目录（无论是否找到，都添加以确保编译通过）
include_directories(${TENSORRT_ROOT}/include)

# 设置全局包含目录，所有源文件都可以访问这些头文件
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include  # 项目自定义头文件
    ${VISP_INCLUDE_DIRS}                  # VISP 头文件
    ${OpenCV_INCLUDE_DIRS}                # OpenCV 头文件
)

# 添加 CUDA 包含目录（如果找到）
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# ------------------------------------------------------------------------------
# 4. 源文件设置
# ------------------------------------------------------------------------------

# 递归查找所有源文件
# GLOB_RECURSE 会递归查找 src 目录及其子目录中的所有 .cpp 文件
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# 打印找到的源文件数量，用于验证
list(LENGTH SOURCES SOURCES_COUNT)
message(STATUS "找到 ${SOURCES_COUNT} 个源文件")

# ------------------------------------------------------------------------------
# 5. 构建配置
# ------------------------------------------------------------------------------

# 构建可执行文件
# IBVS_Teleoperation 是可执行文件名称，${SOURCES} 是所有源文件
add_executable(IBVS_Teleoperation ${SOURCES})

# ------------------------------------------------------------------------------
# 6. 链接配置
# ------------------------------------------------------------------------------

# 链接必需的库
target_link_libraries(IBVS_Teleoperation
    ${VISP_LIBRARIES}    # VISP 库
    ${OpenCV_LIBS}       # OpenCV 库
)

# 链接 CUDA 库（如果找到）
if(CUDA_FOUND)
    target_link_libraries(IBVS_Teleoperation
        ${CUDA_LIBRARIES}      # CUDA 运行时库
        ${CUDA_CUDART_LIBRARY} # CUDA C 运行时库
    )
endif()

# 链接 TensorRT 库（如果找到）
if(NVINFER_LIBRARY)
    target_link_libraries(IBVS_Teleoperation
        ${NVINFER_LIBRARY}           # TensorRT 核心库
        ${NVPARSERS_LIBRARY}         # TensorRT 解析器库
        ${NVONNXPARSERS_LIBRARY}     # ONNX 解析器库
        ${NVINFER_PLUGIN_LIBRARY}    # TensorRT 插件库
    )
endif()