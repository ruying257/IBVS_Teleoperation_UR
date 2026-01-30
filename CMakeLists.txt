cmake_minimum_required(VERSION 3.10)

# 项目名称
project(IBVS_Teleoperation_UR CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找依赖库
set(VISP_DIR "/home/kun/visp-ws/visp-build")

# 打印 VISP_DIR 路径
message(STATUS "VISP_DIR: ${VISP_DIR}")

# 检查 VISP_DIR 是否存在
if(EXISTS ${VISP_DIR})
    message(STATUS "VISP_DIR 存在")
else()
    message(WARNING "VISP_DIR 不存在: ${VISP_DIR}")
endif()

# 查找 VISP 库
find_package(VISP REQUIRED COMPONENTS 
    visp_core 
    visp_gui 
    visp_visual_features 
    visp_robot 
    visp_sensor 
    visp_detection 
    visp_vs 
)

# 打印 VISP 相关信息
message(STATUS "VISP 版本: ${VISP_VERSION}")
message(STATUS "VISP 包含目录: ${VISP_INCLUDE_DIRS}")
message(STATUS "VISP 库: ${VISP_LIBRARIES}")

find_package(OpenCV REQUIRED)

# 查找 CUDA
find_package(CUDA QUIET)
if(CUDA_FOUND)
    message(STATUS "CUDA 版本: ${CUDA_VERSION}")
    message(STATUS "CUDA 包含目录: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA 库: ${CUDA_LIBRARIES}")
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${VISP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# 添加 CUDA 包含目录（如果找到）
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# 查找源文件
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# 构建可执行文件
add_executable(IBVS_Teleoperation ${SOURCES})

# 链接库
target_link_libraries(IBVS_Teleoperation
    ${VISP_LIBRARIES}
    ${OpenCV_LIBS}
)

# 添加 CUDA 库（如果找到）
if(CUDA_FOUND)
    target_link_libraries(IBVS_Teleoperation
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
    )
endif()

# 添加 TensorRT 库（如果找到）
if(NVINFER_LIBRARY)
    target_link_libraries(IBVS_Teleoperation
        ${NVINFER_LIBRARY}
        ${NVPARSERS_LIBRARY}
        ${NVONNXPARSERS_LIBRARY}
        ${NVINFER_PLUGIN_LIBRARY}
    )
endif()

# 安装目标
install(TARGETS IBVS_Teleoperation
    RUNTIME DESTINATION bin
)
