# YOLO 检测器测试程序设计

## 1. 测试程序目标

* 专门测试 YOLO 检测器的效果

* 支持从相机或图像文件获取输入

* 显示检测结果和性能统计

* 提供配置选项来调整测试参数

## 2. 测试程序结构

### 2.1 文件结构

```
├── test_yolo_detector.cpp    # 主测试程序
└── CMakeLists.txt            # 测试程序的 CMake 配置
```

### 2.2 核心功能模块

#### 2.2.1 命令行参数解析

* 支持指定 YOLO 模型路径

* 支持选择输入源（相机或图像文件）

* 支持设置检测参数（置信度阈值、NMS 阈值等）

#### 2.2.2 输入源管理

* 相机输入：RealSense D405 相机

* 图像输入：支持从文件加载图像

* 视频输入：支持从视频文件加载视频

#### 2.2.3 YOLO 检测器封装

* 封装 TensorRT\_detection 类

* 提供统一的检测接口

* 处理模型加载和初始化

#### 2.2.4 结果显示和统计

* 绘制检测边界框和类别信息

* 显示性能统计（FPS、处理时间等）

* 支持保存检测结果到文件

## 3. 测试程序实现

### 3.1 主测试程序 (test\_yolo\_detector.cpp)

#### 3.1.1 包含头文件

```cpp
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

#include <visp3/core/vpImage.h>
#include <visp3/core/vpDisplay.h>
#include <visp3/core/vpDisplayX.h>
#include <visp3/sensor/vpRealSense2.h>

#include "TensorRT_detection.h"
```

#### 3.1.2 命令行参数解析

```cpp
struct TestConfig {
    std::string model_path = "";
    std::string input_source = "camera";
    std::string input_file = "";
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    bool save_results = false;
    std::string output_dir = "results";
};

TestConfig parseCommandLine(int argc, char** argv);
```

#### 3.1.3 主函数

```cpp
int main(int argc, char** argv) {
    try {
        // 解析命令行参数
        TestConfig config = parseCommandLine(argc, argv);
        
        // 初始化 YOLO 检测器
        TensorRT_detection detector(config.model_path);
        
        // 初始化输入源
        // ...
        
        // 主测试循环
        // ...
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}
```

### 3.2 CMake 配置 (CMakeLists.txt)

#### 3.2.1 基本配置

```cmake
cmake_minimum_required(VERSION 3.10)
project(test_yolo_detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

#### 3.2.2 依赖配置

```cmake
# 查找 VISP
find_package(VISP REQUIRED COMPONENTS visp_core visp_gui visp_sensor)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${VISP_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
```

#### 3.2.3 目标配置

```cmake
# 构建测试程序
add_executable(test_yolo_detector test_yolo_detector.cpp)

# 链接库
target_link_libraries(test_yolo_detector
    ${VISP_LIBRARIES}
    ${OpenCV_LIBS}
)
```

## 4. 测试程序使用方法

### 4.1 编译测试程序

```bash
cd IBVS_Teleoperation_UR
mkdir -p test/build
cd test/build
cmake ..
make -j$(nproc)
```

### 4.2 运行测试程序

#### 4.2.1 从相机测试

```bash
./test_yolo_detector --model /path/to/model.trt --source camera
```

#### 4.2.2 从图像文件测试

```bash
./test_yolo_detector --model /path/to/model.trt --source image --input /path/to/image.jpg
```

#### 4.2.3 从视频文件测试

```bash
./test_yolo_detector --model /path/to/model.trt --source video --input /path/to/video.mp4
```

### 4.3 命令行参数

* `--model`：YOLO 模型路径

* `--source`：输入源（camera/image/video）

* `--input`：输入文件路径（当 source 为 image 或 video 时使用）

* `--confidence`：置信度阈值

* `--nms`：NMS 阈值

* `--save`：保存检测结果

* `--output`：输出目录

## 5. 测试结果分析

### 5.1 性能指标

* **FPS**：每秒处理的帧数

* **处理时间**：每帧的平均处理时间

* **内存占用**：GPU 内存使用情况

### 5.2 检测效果评估

* **准确率**：正确检测的目标数量

* **召回率**：检测到的目标占总目标的比例

* **定位精度**：边界框的定位精度

* **类别准确性**：类别识别的准确性

## 6. 扩展功能

### 6.1 批量测试

* 支持批量处理多个图像文件

* 生成测试报告

### 6.2 模型比较

* 支持比较不同 YOLO 模型的性能

* 生成性能对比图表

### 6.3 可视化工具

* 实时显示检测结果

* 支持调整检测参数并实时查看效果

## 7. 总结

本测试程序将为 YOLO 检测器提供一个全面的测试环境，帮助用户评估检测器的性能和效果，为实际应用中的参数调整提供参考。
