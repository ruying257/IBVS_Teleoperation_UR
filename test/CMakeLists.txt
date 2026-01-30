cmake_minimum_required(VERSION 3.10)
project(test_yolo_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找 VISP
set(VISP_DIR "/home/kun/visp-ws/visp-build")
if(EXISTS ${VISP_DIR})
    message(STATUS "VISP_DIR 存在: ${VISP_DIR}")
else()
    message(STATUS "VISP_DIR 不存在，使用系统默认路径")
endif()

find_package(VISP REQUIRED COMPONENTS visp_core visp_gui visp_sensor visp_io)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 查找 CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA 版本: ${CUDA_VERSION}")
else()
    message(WARNING "CUDA 未找到，YOLO 检测将不可用")
endif()

# 查找 TensorRT
find_library(NVINFER_LIBRARY nvinfer)
find_library(NVPARSERS_LIBRARY nvparsers)
find_library(NVONNXPARSERS_LIBRARY nvonnxparser)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin)

if(NVINFER_LIBRARY)
    message(STATUS "TensorRT 库: ${NVINFER_LIBRARY}")
    add_definitions(-DHAVE_TENSORRT)
else()
    message(WARNING "TensorRT 未找到，YOLO 检测将不可用")
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${VISP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 添加 CUDA 包含目录（如果找到）
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# 构建测试程序
add_executable(test_yolo_detector test_yolo_detector.cpp)

# 链接库
target_link_libraries(test_yolo_detector
    ${VISP_LIBRARIES}
    ${OpenCV_LIBS}
)

# 添加 CUDA 库（如果找到）
if(CUDA_FOUND)
    target_link_libraries(test_yolo_detector
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
    )
endif()

# 添加 TensorRT 库（如果找到）
if(NVINFER_LIBRARY)
    target_link_libraries(test_yolo_detector
        ${NVINFER_LIBRARY}
        ${NVPARSERS_LIBRARY}
        ${NVONNXPARSERS_LIBRARY}
        ${NVINFER_PLUGIN_LIBRARY}
    )
endif()
