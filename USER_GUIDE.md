# 使用说明文档

## 1. 项目概述

**IBVS_Teleoperation_UR** 是一个基于视觉伺服的机器人遥操作系统，专为 Universal Robots 机器人设计，集成了实时目标检测、视觉伺服控制和遥操作功能。

### 1.1 支持的硬件
- **机器人**：Universal Robots 系列机器人（如 UR5、UR10 等）
- **相机**：Intel RealSense D405
- **计算机**：支持 CUDA 的 NVIDIA GPU（推荐，用于 YOLO 检测加速）

## 2. 环境搭建

### 2.1 依赖安装

#### 2.1.1 核心依赖
- **VISP 3.7.1**
  ```bash
  # 从源码编译安装 VISP
  git clone https://github.com/lagadic/visp.git
  cd visp
  mkdir build && cd build
  cmake .. -DBUILD_DEMOS=OFF -DBUILD_EXAMPLES=OFF
  make -j$(nproc)
  sudo make install
  ```

- **OpenCV**
  ```bash
  sudo apt-get install libopencv-dev
  ```

- **Eigen3**
  ```bash
  sudo apt-get install libeigen3-dev
  ```

- **RealSense SDK**
  ```bash
  sudo apt-get install librealsense2-dev
  ```

#### 2.1.2 可选依赖（用于 YOLO 检测）
- **CUDA**
  - 从 NVIDIA 官网下载并安装对应版本的 CUDA

- **TensorRT**
  - 从 NVIDIA 官网下载并安装对应版本的 TensorRT

### 2.2 项目编译

1. **克隆项目**
   ```bash
   git clone <项目地址>
   cd IBVS_Teleoperation_UR
   ```

2. **配置 CMake**
   ```bash
   mkdir build && cd build
   cmake ..
   ```

3. **编译项目**
   ```bash
   make -j$(nproc)
   ```

### 2.3 编译选项

- **CUDA 支持**：自动检测，无需额外配置
- **TensorRT 支持**：自动检测，无需额外配置
- **相机支持**：默认启用 RealSense D405 支持

## 3. 系统配置

### 3.1 配置文件

配置参数通过 `AppConfig` 结构体在代码中设置，位于 `include/AppConfig.h` 文件中。主要配置项包括：

#### 3.1.1 相机参数
- `camera_width`：相机宽度（默认 640）
- `camera_height`：相机高度（默认 480）

#### 3.1.2 AprilTag 参数
- `tag_size`：AprilTag 标签大小（单位：米）
- `tag_quad_decimate`：检测精度（默认 1.0）

#### 3.1.3 视觉伺服参数
- `adaptive_gain`：是否使用自适应增益（默认 true）
- `display_tag`：是否显示 AprilTag 检测结果（默认 true）

#### 3.1.4 遥操作参数
- `teleop_linear_velocity`：线性速度（默认 0.1 m/s）
- `teleop_angular_velocity`：角速度（默认 0.5 rad/s）

#### 3.1.5 YOLO 检测参数
- `yolo_model_path`：YOLO 模型路径（默认 ""）

### 3.2 配置示例

```cpp
// include/AppConfig.h
struct AppConfig {
    // 相机参数
    int camera_width = 640;
    int camera_height = 480;
    
    // AprilTag 参数
    double tag_size = 0.05;  // 5cm 标签
    float tag_quad_decimate = 1.0f;
    
    // 视觉伺服参数
    bool adaptive_gain = true;
    bool display_tag = true;
    
    // 遥操作参数
    double teleop_linear_velocity = 0.1;
    double teleop_angular_velocity = 0.5;
    
    // YOLO 检测参数
    std::string yolo_model_path = "";
};
```

## 4. 运行系统

### 4.1 基本运行

1. **连接硬件**
   - 连接 Universal Robots 机器人
   - 连接 RealSense D405 相机

2. **启动系统**
   ```bash
   cd build
   ./IBVS_Teleoperation
   ```

### 4.2 运行模式

系统启动后会自动进入 **视觉伺服模式**，可以通过键盘切换到 **遥操作模式**。

#### 4.2.1 视觉伺服模式
- **功能**：自动追踪 AprilTag 目标
- **控制**：无需手动控制，系统自动执行

#### 4.2.2 遥操作模式
- **功能**：通过键盘控制机器人运动
- **控制按键**：
  - **平移控制**：WASD 键控制前后左右
  - **旋转控制**：QE 键控制旋转
  - **Z 轴控制**：RF 键控制上下
  - **速度模式**：Shift 键切换快速/精细模式
  - **急停**：Space 键急停
  - **退出**：Q 键退出

### 4.3 YOLO 目标检测

#### 4.3.1 启用 YOLO 检测
1. **准备模型文件**：获取 `.trt` 格式的 YOLO 模型
2. **设置模型路径**：在 `SystemController.cpp` 中设置模型路径
   ```cpp
   std::string yolo_model_path = "path/to/your/model.trt";
   ```
3. **重新编译**：
   ```bash
   cd build && make -j$(nproc)
   ```

#### 4.3.2 查看检测结果
- **实时显示**：检测结果会实时绘制在相机图像上
- **控制台输出**：显示检测统计信息和性能数据

## 5. 系统调试

### 5.1 日志输出

系统运行时会在控制台输出以下信息：
- **初始化信息**：各模块初始化状态
- **检测结果**：AprilTag 和 YOLO 检测结果
- **控制信息**：视觉伺服控制指令
- **遥操作信息**：遥操作控制指令
- **错误信息**：系统错误和异常

### 5.2 常见问题

#### 5.2.1 相机连接失败
- **症状**：系统启动时提示相机连接失败
- **解决方案**：
  - 检查相机是否正确连接
  - 检查 RealSense SDK 是否正确安装
  - 尝试重新插拔相机

#### 5.2.2 机器人连接失败
- **症状**：系统启动时提示机器人连接失败
- **解决方案**：
  - 检查机器人是否正确连接
  - 检查机器人是否处于远程控制模式
  - 检查网络连接

#### 5.2.3 YOLO 检测不工作
- **症状**：系统启动时提示 TensorRT 未找到
- **解决方案**：
  - 检查 CUDA 和 TensorRT 是否正确安装
  - 确保模型路径正确
  - 检查 GPU 是否支持 CUDA

#### 5.2.4 视觉伺服不稳定
- **症状**：视觉伺服控制不稳定，机器人抖动
- **解决方案**：
  - 调整视觉伺服增益参数
  - 确保相机固定牢固
  - 确保 AprilTag 标签清晰可见

## 6. 高级使用

### 6.1 自定义目标检测

#### 6.1.1 添加新的目标类别
1. **修改 YOLO 模型**：训练支持新类别的 YOLO 模型
2. **更新类别名称**：在 `TensorRT_detection.cpp` 中更新类别名称列表
   ```cpp
   m_class_names.push_back("new_class");
   ```

#### 6.1.2 自定义检测逻辑
- 扩展 `TensorRT_detection` 类，添加自定义检测方法
- 或创建新的检测类，集成到系统中

### 6.2 自定义控制算法

#### 6.2.1 视觉伺服算法
- 修改 `SystemController.cpp` 中的 `process_ibvs()` 方法
- 调整视觉伺服参数和控制逻辑

#### 6.2.2 遥操作算法
- 修改 `RobotTeleoperation.cpp` 中的控制逻辑
- 添加新的控制模式和功能

### 6.3 多机器人协同

- 通过扩展 `SystemController` 类，添加多机器人支持
- 实现机器人之间的通信和协作逻辑

## 7. 安全注意事项

### 7.1 安全操作

1. **使用前测试**：在安全环境中测试系统
2. **定期校准**：定期校准相机和机器人
3. **监控系统**：实时监控系统状态
4. **紧急停止**：熟悉急停操作方法

### 7.2 安全限制

- **速度限制**：系统内置机器人速度限制
- **工作空间限制**：确保机器人在安全工作空间内操作
- **碰撞检测**：建议添加额外的碰撞检测系统

## 8. 总结

**IBVS_Teleoperation_UR** 系统提供了一个完整的机器人视觉伺服和遥操作解决方案，通过集成先进的视觉检测和控制算法，为机器人操作提供了直观、高效的控制方式。

系统设计考虑了实时性能和可靠性，适合在工业环境中使用。同时，通过模块化的设计，为系统扩展和功能定制提供了灵活性。

## 9. 联系方式

如有问题或建议，请联系项目维护人员。

- **项目地址**：<项目地址>
- **维护人员**：<维护人员信息>
- **电子邮件**：<电子邮件>
