// ============================================================================
// YOLO 检测器测试程序
// 功能：测试 TensorRT 加速的 YOLO 目标检测模型
// 支持输入源：相机、单张图像、视频文件
// ============================================================================

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <getopt.h>
#include <filesystem>

// ViSP 视觉库头文件
#include <visp/vpImage.h>           // 图像类
#include <visp/vpDisplay.h>         // 显示基础类
#include <visp/vpDisplayX.h>        // X 窗口显示
#include <visp/vpRealSense2.h>      // RealSense 相机接口
#include <visp/vpIoTools.h>         // IO 工具
#include <visp/vpImageConvert.h>    // 图像转换
#include <visp/vpImageIo.h>         // 图像 IO 操作

// TensorRT 检测模块
#include "TensorRT_detection.h"

// OpenCV 视频处理
#include <opencv2/videoio.hpp>
#include <visp/vpImageIo.h>

// 文件系统命名空间
namespace fs = std::filesystem;

// ============================================================================
// 测试配置结构体
// 用于存储命令行参数和测试配置信息
// ============================================================================
struct TestConfig {
    std::string model_path = "../../models/best.trt";   // TensorRT 模型路径
    std::string source = "camera";                      // 输入源类型：camera, image, video
    std::string input_file = "";                        // 输入文件路径（图像或视频）
    float confidence_threshold = 0.5f;                  // 置信度阈值
    float nms_threshold = 0.4f;                         // NMS（非极大值抑制）阈值
    bool save_results = false;                          // 是否保存检测结果
    std::string output_dir = "results";                 // 输出结果目录
    bool show_stats = true;                             // 是否显示性能统计
    int camera_width = 640;                             // 相机宽度
    int camera_height = 480;                            // 相机高度
};

// ============================================================================
// 命令行参数解析函数
// 功能：解析命令行参数并填充 TestConfig 结构体
// 参数：
//   argc - 命令行参数数量
//   argv - 命令行参数数组
// 返回值：
//   填充后的 TestConfig 结构体
// ============================================================================
TestConfig parseCommandLine(int argc, char** argv) {
    TestConfig config;

    // 长选项定义
    const struct option long_options[] = {
        {"model", required_argument, nullptr, 'm'},
        {"source", required_argument, nullptr, 's'},
        {"input", required_argument, nullptr, 'i'},
        {"confidence", required_argument, nullptr, 'c'},
        {"nms", required_argument, nullptr, 'n'},
        {"save", no_argument, nullptr, 'S'},
        {"output", required_argument, nullptr, 'o'},
        {"stats", no_argument, nullptr, 't'},
        {"width", required_argument, nullptr, 'w'},
        {"height", required_argument, nullptr, 'h'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    // 解析命令行参数
    int opt;
    while ((opt = getopt_long(argc, argv, "m:s:i:c:n:So:tw:h:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                config.model_path = optarg;      // 设置模型路径
                break;
            case 's':
                config.source = optarg;          // 设置输入源类型
                break;
            case 'i':
                config.input_file = optarg;      // 设置输入文件路径
                break;
            case 'c':
                config.confidence_threshold = std::stof(optarg);  // 设置置信度阈值
                break;
            case 'n':
                config.nms_threshold = std::stof(optarg);         // 设置 NMS 阈值
                break;
            case 'S':
                config.save_results = true;      // 启用结果保存
                break;
            case 'o':
                config.output_dir = optarg;      // 设置输出目录
                break;
            case 't':
                config.show_stats = true;        // 启用统计信息显示
                break;
            case 'w':
                config.camera_width = std::stoi(optarg);  // 设置相机宽度
                break;
            case 'h':
                config.camera_height = std::stoi(optarg); // 设置相机高度
                break;
            case '?':
            default:
                // 显示帮助信息
                std::cout << "Usage: test_yolo_detector [options]" << std::endl;
                std::cout << "Options:" << std::endl;
                std::cout << "  --model, -m <path>    YOLO model path" << std::endl;
                std::cout << "  --source, -s <type>   Input source (camera/image/video)" << std::endl;
                std::cout << "  --input, -i <path>    Input file path" << std::endl;
                std::cout << "  --confidence, -c <val> Confidence threshold" << std::endl;
                std::cout << "  --nms, -n <val>       NMS threshold" << std::endl;
                std::cout << "  --save, -S            Save detection results" << std::endl;
                std::cout << "  --output, -o <path>   Output directory" << std::endl;
                std::cout << "  --stats, -t           Show performance stats" << std::endl;
                std::cout << "  --width, -w <val>     Camera width" << std::endl;
                std::cout << "  --height, -h <val>    Camera height" << std::endl;
                std::exit(EXIT_FAILURE);
        }
    }

    // 验证参数有效性
    if (config.model_path.empty()) {
        std::cerr << "Error: Model path is required" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // 对于图像和视频输入源，必须提供输入文件路径
    if ((config.source == "image" || config.source == "video") && config.input_file.empty()) {
        std::cerr << "Error: Input file path is required for image/video source" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return config;
}

// ============================================================================
// 性能统计类
// 功能：统计检测过程中的性能指标，如 FPS、平均处理时间等
// ============================================================================
class PerformanceStats {
public:
    /**
     * @brief 构造函数
     * 初始化统计计数器和时间
     */
    PerformanceStats() : frame_count(0), total_time(0) {}

    /**
     * @brief 开始一帧的计时
     * 记录当前时间作为帧开始时间
     */
    void startFrame() {
        frame_start = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief 结束一帧的计时
     * 计算帧处理时间并累加到总时间
     */
    void endFrame() {
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        total_time += frame_time;
        frame_count++;
    }

    /**
     * @brief 计算 FPS（每秒帧数）
     * @return FPS 值
     */
    double getFPS() const {
        if (total_time == 0) return 0;
        return (frame_count * 1000.0) / total_time;
    }

    /**
     * @brief 计算平均处理时间
     * @return 平均处理时间（毫秒）
     */
    double getAverageTime() const {
        if (frame_count == 0) return 0;
        return total_time / (double)frame_count;
    }

    /**
     * @brief 重置统计数据
     */
    void reset() {
        frame_count = 0;
        total_time = 0;
    }

    /**
     * @brief 获取处理的总帧数
     * @return 总帧数
     */
    int getFrameCount() const { return frame_count; }

private:
    int frame_count;                     // 处理的总帧数
    long long total_time;                // 总处理时间（毫秒）
    std::chrono::high_resolution_clock::time_point frame_start;  // 帧开始时间
};

// ============================================================================
// 保存检测结果
// 功能：将检测结果保存为图像和文本文件
// 参数：
//   image - 输入图像
//   result - 检测结果
//   output_dir - 输出目录
//   frame_idx - 帧索引
// ============================================================================
void saveDetectionResult(const vpImage<unsigned char>& image, const DetectionResult& result, 
                        const std::string& output_dir, int frame_idx) {
    // 创建输出目录（如果不存在）
    fs::create_directories(output_dir);

    // 保存图像
    std::string image_path = output_dir + "/frame_" + std::to_string(frame_idx) + ".jpg";
    vpImage<vpRGBa> rgb_image;
    vpImageConvert::convert(image, rgb_image);  // 转换为 RGB 图像
    vpImageIo::write(rgb_image, image_path);    // 写入文件

    // 保存检测结果到文本文件
    std::string result_path = output_dir + "/detections_" + std::to_string(frame_idx) + ".txt";
    std::ofstream result_file(result_path);
    if (result_file.is_open()) {
        // 写入结果头部信息
        result_file << "# Detection results" << std::endl;
        result_file << "# Processing time: " << result.processing_time_ms << " ms" << std::endl;
        result_file << "# Objects detected: " << result.bolts.size() << std::endl;
        result_file << "# Format: class_id confidence x y width height class_name" << std::endl;

        // 写入每个检测对象的信息
        for (const auto& bolt : result.bolts) {
            result_file << bolt.class_id << " " 
                       << bolt.confidence << " "
                       << bolt.bounding_box.getTopLeft().get_u() << " "
                       << bolt.bounding_box.getTopLeft().get_v() << " "
                       << bolt.bounding_box.getWidth() << " "
                       << bolt.bounding_box.getHeight() << " "
                       << bolt.class_name << std::endl;
        }
        result_file.close();
    }
}

/**
 * @brief 在图像上显示YOLO检测的性能统计信息和结果
 * 
 * 该函数将检测过程中的性能指标和检测结果以文本形式绘制在输入图像上，
 * 方便用户实时监控系统运行状态和检测效果。所有统计信息将以红色文本
 * 显示在图像左上角，按行排列。
 * 
 * @param[in,out] image 输入的ViSP灰度图像，函数会在其上绘制统计信息
 * @param[in] stats 性能统计对象，包含FPS、平均处理时间等信息
 * @param[in] result YOLO检测结果对象，包含检测到的螺栓信息和处理时间
 * 
 * @note 该函数使用ViSP的vpDisplay::displayText方法进行文本绘制，
 *       文本位置从左上角(20,20)开始，每行间隔20像素。
 */
void displayStats(vpImage<unsigned char>& image, const PerformanceStats& stats, 
                 const DetectionResult& result) {
    std::stringstream ss;
    
    // 显示 FPS
    ss << "FPS: " << std::fixed << std::setprecision(1) << stats.getFPS();
    vpDisplay::displayText(image, 20, 20, ss.str(), vpColor::red);

    // 显示平均处理时间
    ss.str("");
    ss << "Avg time: " << std::fixed << std::setprecision(1) << stats.getAverageTime() << " ms";
    vpDisplay::displayText(image, 40, 20, ss.str(), vpColor::red);

    // 显示检测到的对象数量
    ss.str("");
    ss << "Objects: " << result.bolts.size();
    vpDisplay::displayText(image, 60, 20, ss.str(), vpColor::red);

    // 显示当前帧处理时间
    ss.str("");
    ss << "Process time: " << result.processing_time_ms << " ms";
    vpDisplay::displayText(image, 80, 20, ss.str(), vpColor::red);
}

// ============================================================================
// 主测试函数
// 功能：主程序入口，执行 YOLO 检测器测试
// 参数：
//   argc - 命令行参数数量
//   argv - 命令行参数数组
// 返回值：
//   EXIT_SUCCESS 或 EXIT_FAILURE
// ============================================================================
int main(int argc, char** argv) {
    try {
        // 解析命令行参数
        TestConfig config = parseCommandLine(argc, argv);
        
        // 打印测试配置信息
        std::cout << "=== YOLO 检测器测试程序 ===" << std::endl;
        std::cout << "模型路径: " << config.model_path << std::endl;
        std::cout << "输入源: " << config.source << std::endl;
        if (!config.input_file.empty()) {
            std::cout << "输入文件: " << config.input_file << std::endl;
        }
        std::cout << "置信度阈值: " << config.confidence_threshold << std::endl;
        std::cout << "NMS 阈值: " << config.nms_threshold << std::endl;
        std::cout << "保存结果: " << (config.save_results ? "是" : "否") << std::endl;
        if (config.save_results) {
            std::cout << "输出目录: " << config.output_dir << std::endl;
        }
        std::cout << "显示统计: " << (config.show_stats ? "是" : "否") << std::endl;
        std::cout << "========================" << std::endl;

        // 初始化 YOLO 检测器
        std::cout << "初始化 YOLO 检测器..." << std::endl;
        TensorRT_detection detector(config.model_path);
        std::cout << "YOLO 检测器初始化完成" << std::endl;

        // 初始化变量
        vpImage<unsigned char> I(config.camera_height, config.camera_width);  // 输入图像
        vpDisplayX display;                                                   // 显示窗口
        bool running = true;                                                  // 运行标志
        int frame_idx = 0;                                                    // 帧索引
        PerformanceStats stats;                                               // 性能统计对象

        // 根据输入源类型执行不同的处理逻辑
        if (config.source == "camera") {
            // 相机输入处理
            std::cout << "初始化相机..." << std::endl;
            vpRealSense2 rs;                                  // RealSense 相机对象
            rs2::config rs_config;                             // 相机配置
            
            // 配置相机流
            rs_config.enable_stream(RS2_STREAM_COLOR, config.camera_width, config.camera_height, RS2_FORMAT_RGB8, 30);
            rs_config.enable_stream(RS2_STREAM_DEPTH, config.camera_width, config.camera_height, RS2_FORMAT_Z16, 30);
            rs.open(rs_config);                               // 打开相机
            std::cout << "相机初始化完成" << std::endl;

            // 初始化显示窗口
            display.init(I, 10, 10, "YOLO TEST");

            // 主测试循环
            std::cout << "开始测试... (按 'q' 退出)" << std::endl;
            while (running) {
                stats.startFrame();  // 开始帧计时

                // 获取相机图像
                rs.acquire(I);
                vpDisplay::display(I);

                // 准备检测数据
                FrameData frame;
                vpImage<vpRGBa> rgb_image;
                vpImageConvert::convert(I, rgb_image);  // 转换为 RGB 图像
                frame.image = rgb_image;
                frame.timestamp = 0;

                // 执行推理
                DetectionResult detection_result;
                detector.infer_trtmodel(frame, detection_result);

                // 绘制检测结果
                if (detection_result.success) {
                    for (const auto& bolt : detection_result.bolts) {
                        // 绘制边界框
                        vpDisplay::displayRectangle(I, bolt.bounding_box, vpColor::green, false, 2);
                        // 绘制标签
                        std::string label = bolt.class_name + " " + std::to_string(bolt.confidence);
                        vpDisplay::displayText(I, bolt.bounding_box.getTopLeft() + vpImagePoint(-10, -10), 
                                              label, vpColor::green);
                    }
                }

                // 显示性能统计
                if (config.show_stats) {
                    displayStats(I, stats, detection_result);
                }

                // 保存结果（每10帧保存一次）
                if (config.save_results && frame_idx % 10 == 0) {
                    saveDetectionResult(I, detection_result, config.output_dir, frame_idx);
                }

                stats.endFrame();  // 结束帧计时
                frame_idx++;

                // 显示图像
                vpDisplay::flush(I);

                // 检查键盘输入
                if (vpDisplay::getKeyboardEvent(I, false)) {
                    char key = vpDisplay::getKeyboardEvent(I);
                    if (key == 'q' || key == 'Q') {
                        running = false;  // 按 'q' 退出
                    }
                }
            }

        } else if (config.source == "image") {
            // 单张图像处理
            std::cout << "加载图像: " << config.input_file << std::endl;
            vpImage<vpRGBa> rgb_image;
            vpImageIo::read(rgb_image, config.input_file);  // 读取图像
            vpImageConvert::convert(rgb_image, I);         // 转换为灰度图像
            display.init(I, 10, 10, "YOLO 检测器测试");   // 初始化显示窗口

            stats.startFrame();  // 开始计时

            // 准备检测数据
            FrameData frame;
            frame.image = rgb_image;
            frame.timestamp = 0;

            // 执行推理
            DetectionResult detection_result;
            detector.infer_trtmodel(frame, detection_result);

            // 绘制检测结果
            if (detection_result.success) {
                for (const auto& bolt : detection_result.bolts) {
                    vpDisplay::displayRectangle(I, bolt.bounding_box, vpColor::green, false, 2);
                    std::string label = bolt.class_name + " " + std::to_string(bolt.confidence);
                    vpDisplay::displayText(I, bolt.bounding_box.getTopLeft() + vpImagePoint(-10, -10), 
                                          label, vpColor::green);
                }
            }

            // 显示性能统计
            if (config.show_stats) {
                displayStats(I, stats, detection_result);
            }

            stats.endFrame();  // 结束计时

            // 保存结果
            if (config.save_results) {
                saveDetectionResult(I, detection_result, config.output_dir, 0);
            }

            // 显示图像
            vpDisplay::display(I);
            vpDisplay::flush(I);
            std::cout << "按任意键退出..." << std::endl;
            vpDisplay::getKeyboardEvent(I);  // 等待键盘输入

        } else if (config.source == "video") {
            // 视频处理
            std::cout << "加载视频: " << config.input_file << std::endl;
            cv::VideoCapture cap(config.input_file);  // 打开视频文件
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open video file" << std::endl;
                return EXIT_FAILURE;
            }

            // 获取视频尺寸
            int video_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int video_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            I.resize(video_height, video_width);  // 调整图像大小
            display.init(I, 10, 10, "YOLO 检测器测试");  // 初始化显示窗口

            std::cout << "开始测试... (按 'q' 退出)" << std::endl;
            while (running) {
                cv::Mat frame;
                if (!cap.read(frame)) {
                    break;  // 视频结束
                }

                stats.startFrame();  // 开始帧计时

                // 转换为 ViSP 图像
                vpImage<vpRGBa> rgb_image;
                vpImageConvert::convert(frame, rgb_image);  // OpenCV 转 ViSP
                vpImageConvert::convert(rgb_image, I);      // 转换为灰度
                vpDisplay::display(I);

                // 准备检测数据
                FrameData yolo_frame;
                yolo_frame.image = rgb_image;
                yolo_frame.timestamp = 0;

                // 执行推理
                DetectionResult detection_result;
                detector.infer_trtmodel(yolo_frame, detection_result);

                // 绘制检测结果
                if (detection_result.success) {
                    for (const auto& bolt : detection_result.bolts) {
                        vpDisplay::displayRectangle(I, bolt.bounding_box, vpColor::green, false, 2);
                        std::string label = bolt.class_name + " " + std::to_string(bolt.confidence);
                        vpDisplay::displayText(I, bolt.bounding_box.getTopLeft() + vpImagePoint(-10, -10), 
                                              label, vpColor::green);
                    }
                }

                // 显示性能统计
                if (config.show_stats) {
                    displayStats(I, stats, detection_result);
                }

                // 保存结果（每10帧保存一次）
                if (config.save_results && frame_idx % 10 == 0) {
                    saveDetectionResult(I, detection_result, config.output_dir, frame_idx);
                }

                stats.endFrame();  // 结束帧计时
                frame_idx++;

                // 显示图像
                vpDisplay::flush(I);

                // 检查键盘输入
                if (vpDisplay::getKeyboardEvent(I, false)) {
                    char key = vpDisplay::getKeyboardEvent(I);
                    if (key == 'q' || key == 'Q') {
                        running = false;  // 按 'q' 退出
                    }
                }
            }

        } else {
            // 无效的输入源类型
            std::cerr << "Error: Invalid source type" << std::endl;
            return EXIT_FAILURE;
        }

        // 显示最终统计信息
        std::cout << "\n=== 测试完成 ===" << std::endl;
        std::cout << "总帧数: " << stats.getFrameCount() << std::endl;
        std::cout << "平均 FPS: " << std::fixed << std::setprecision(1) << stats.getFPS() << std::endl;
        std::cout << "平均处理时间: " << std::fixed << std::setprecision(1) << stats.getAverageTime() << " ms" << std::endl;
        std::cout << "================" << std::endl;

    } catch (const std::exception& e) {
        // 异常处理
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
