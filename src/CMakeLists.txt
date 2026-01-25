cmake_minimum_required(VERSION 3.10)

# 项目名称
project(IBVS_Teleoperation_UR CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找依赖库
find_package(VISP REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(CUDA REQUIRED)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${VISP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    # ${CUDA_INCLUDE_DIRS}
)

# 查找源文件
file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

# 构建可执行文件
add_executable(IBVS_Teleoperation ${SOURCES})

# 链接库
target_link_libraries(IBVS_Teleoperation
    ${VISP_LIBRARIES}
    ${OpenCV_LIBS}
    # ${CUDA_LIBRARIES}
    # ${CUDA_CUDART_LIBRARY}
    # nvinfer
)

# 安装目标
install(TARGETS IBVS_Teleoperation
    RUNTIME DESTINATION bin
)
