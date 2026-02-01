# ============================================================================
# CMake 配置文件
# 项目: test_yolo_detector
# 功能: 构建 YOLO 检测器测试程序
# ============================================================================

### 基础配置
cmake_minimum_required(VERSION 3.10)
project(test_yolo_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# 依赖查找
# ============================================================================

# 1. 查找 VISP 库
find_package(VISP REQUIRED COMPONENTS 
    visp_core 
    visp_gui 
    visp_sensor 
    visp_io
)
if(VISP_FOUND)
    message(STATUS "找到 VISP 库: ${VISP_VERSION}")
endif()

# 2. 查找 OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "找到 OpenCV 库: ${OpenCV_VERSION}")
endif()

# 3. 查找 CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "找到 CUDA: ${CUDA_VERSION}")
else()
    message(WARNING "CUDA 未找到，YOLO 检测将不可用")
endif()

# 4. 查找 TensorRT
if(DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_ROOT $ENV{TENSORRT_DIR})
    message(STATUS "找到 TensorRT: ${TENSORRT_ROOT}")
else()
    # 如果没设环境变量，尝试在默认位置找(兼容性)
    set(TENSORRT_ROOT "$ENV{HOME}/TensorRT-8.6")
    message(WARNING "未找到 TensorRT_DIR 环境变量，尝试默认位置: ${TENSORRT_ROOT}")
endif()

# 查找 TensorRT 库
find_library(NVINFER_LIBRARY nvinfer HINTS ${TENSORRT_ROOT}/lib)
find_library(NVPARSERS_LIBRARY nvparsers HINTS ${TENSORRT_ROOT}/lib)
find_library(NVONNXPARSERS_LIBRARY nvonnxparser HINTS ${TENSORRT_ROOT}/lib)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin HINTS ${TENSORRT_ROOT}/lib)

if(NVINFER_LIBRARY)
    message(STATUS "找到 TensorRT 库: ${NVINFER_LIBRARY}")
    add_definitions(-DHAVE_TENSORRT)
else()
    message(WARNING "TensorRT 未找到，YOLO 检测将不可用")
endif()

# ============================================================================
# 包含目录设置
# ============================================================================

# 项目包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

# 依赖库包含目录
include_directories(
    ${VISP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# TensorRT 包含目录
if(NVINFER_LIBRARY)
    include_directories(${TENSORRT_ROOT}/include)
endif()

# CUDA 包含目录（如果找到）
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# ============================================================================
# 源文件设置
# ============================================================================

# 获取 TensorRT_detection 的源文件
set(TENSORRT_DETECTION_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/../src/TensorRT_detection.cpp)

# ============================================================================
# 构建配置
# ============================================================================

# 构建测试程序
add_executable(test_yolo_detector 
    test_yolo_detector.cpp
    ${TENSORRT_DETECTION_SOURCES}
)

# ============================================================================
# 链接配置
# ============================================================================

# 链接基础库
target_link_libraries(test_yolo_detector
    ${VISP_LIBRARIES}
    ${OpenCV_LIBS}
)

# 链接 CUDA 库（如果找到）
if(CUDA_FOUND)
    target_link_libraries(test_yolo_detector
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
    )
endif()

# 链接 TensorRT 库（如果找到）
if(NVINFER_LIBRARY)
    target_link_libraries(test_yolo_detector
        ${NVINFER_LIBRARY}
        ${NVPARSERS_LIBRARY}
        ${NVONNXPARSERS_LIBRARY}
        ${NVINFER_PLUGIN_LIBRARY}
    )
endif()