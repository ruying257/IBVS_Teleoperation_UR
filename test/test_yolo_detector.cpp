#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <getopt.h>
#include <filesystem>

#include <visp/vpImage.h>
#include <visp/vpDisplay.h>
#include <visp/vpDisplayX.h>
#include <visp/vpRealSense2.h>
#include <visp/vpIoTools.h>
#include <visp/vpImageConvert.h>

#include "TensorRT_detection.h"
#include <opencv2/videoio.hpp>
#include <visp/vpImageIo.h>

namespace fs = std::filesystem;

// 测试配置结构体
struct TestConfig {
    std::string model_path = "";
    std::string source = "camera";  // camera, image, video
    std::string input_file = "";
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    bool save_results = false;
    std::string output_dir = "results";
    bool show_stats = true;
    int camera_width = 640;
    int camera_height = 480;
};

// 命令行参数解析
TestConfig parseCommandLine(int argc, char** argv) {
    TestConfig config;

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

    int opt;
    while ((opt = getopt_long(argc, argv, "m:s:i:c:n:So:tw:h:", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'm':
                config.model_path = optarg;
                break;
            case 's':
                config.source = optarg;
                break;
            case 'i':
                config.input_file = optarg;
                break;
            case 'c':
                config.confidence_threshold = std::stof(optarg);
                break;
            case 'n':
                config.nms_threshold = std::stof(optarg);
                break;
            case 'S':
                config.save_results = true;
                break;
            case 'o':
                config.output_dir = optarg;
                break;
            case 't':
                config.show_stats = true;
                break;
            case 'w':
                config.camera_width = std::stoi(optarg);
                break;
            case 'h':
                config.camera_height = std::stoi(optarg);
                break;
            case '?':
            default:
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

    // 验证参数
    if (config.model_path.empty()) {
        std::cerr << "Error: Model path is required" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if ((config.source == "image" || config.source == "video") && config.input_file.empty()) {
        std::cerr << "Error: Input file path is required for image/video source" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return config;
}

// 性能统计类
class PerformanceStats {
public:
    PerformanceStats() : frame_count(0), total_time(0) {}

    void startFrame() {
        frame_start = std::chrono::high_resolution_clock::now();
    }

    void endFrame() {
        auto frame_end = std::chrono::high_resolution_clock::now();
        auto frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end - frame_start).count();
        total_time += frame_time;
        frame_count++;
    }

    double getFPS() const {
        if (total_time == 0) return 0;
        return (frame_count * 1000.0) / total_time;
    }

    double getAverageTime() const {
        if (frame_count == 0) return 0;
        return total_time / (double)frame_count;
    }

    void reset() {
        frame_count = 0;
        total_time = 0;
    }

    int getFrameCount() const { return frame_count; }

private:
    int frame_count;
    long long total_time;  // 毫秒
    std::chrono::high_resolution_clock::time_point frame_start;
};

// 保存检测结果
void saveDetectionResult(const vpImage<unsigned char>& image, const DetectionResult& result, 
                        const std::string& output_dir, int frame_idx) {
    // 创建输出目录
    fs::create_directories(output_dir);

    // 保存图像
    std::string image_path = output_dir + "/frame_" + std::to_string(frame_idx) + ".jpg";
    vpImage<vpRGBa> rgb_image;
    vpImageConvert::convert(image, rgb_image);
    vpImageIo::write(rgb_image, image_path);

    // 保存检测结果
    std::string result_path = output_dir + "/detections_" + std::to_string(frame_idx) + ".txt";
    std::ofstream result_file(result_path);
    if (result_file.is_open()) {
        result_file << "# Detection results" << std::endl;
        result_file << "# Processing time: " << result.processing_time_ms << " ms" << std::endl;
        result_file << "# Objects detected: " << result.bolts.size() << std::endl;
        result_file << "# Format: class_id confidence x y width height class_name" << std::endl;

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

// 显示性能统计
void displayStats(vpImage<unsigned char>& image, const PerformanceStats& stats, 
                 const DetectionResult& result) {
    std::stringstream ss;
    ss << "FPS: " << std::fixed << std::setprecision(1) << stats.getFPS();
    vpDisplay::displayText(image, 20, 20, ss.str(), vpColor::red);

    ss.str("");
    ss << "Avg time: " << std::fixed << std::setprecision(1) << stats.getAverageTime() << " ms";
    vpDisplay::displayText(image, 40, 20, ss.str(), vpColor::red);

    ss.str("");
    ss << "Objects: " << result.bolts.size();
    vpDisplay::displayText(image, 60, 20, ss.str(), vpColor::red);

    ss.str("");
    ss << "Process time: " << result.processing_time_ms << " ms";
    vpDisplay::displayText(image, 80, 20, ss.str(), vpColor::red);
}

// 主测试函数
int main(int argc, char** argv) {
    try {
        // 解析命令行参数
        TestConfig config = parseCommandLine(argc, argv);
        
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

        // 初始化输入源
        vpImage<unsigned char> I(config.camera_height, config.camera_width);
        vpDisplayX display;
        bool running = true;
        int frame_idx = 0;
        PerformanceStats stats;

        if (config.source == "camera") {
            std::cout << "初始化相机..." << std::endl;
            vpRealSense2 rs;
            rs2::config rs_config;
            rs_config.enable_stream(RS2_STREAM_COLOR, config.camera_width, config.camera_height, RS2_FORMAT_RGB8, 30);
            rs_config.enable_stream(RS2_STREAM_DEPTH, config.camera_width, config.camera_height, RS2_FORMAT_Z16, 30);
            rs.open(rs_config);
            std::cout << "相机初始化完成" << std::endl;

            // 显示初始化
            display.init(I, 10, 10, "YOLO 检测器测试");

            // 主测试循环
            std::cout << "开始测试... (按 'q' 退出)" << std::endl;
            while (running) {
                stats.startFrame();

                // 获取相机图像
                rs.acquire(I);
                vpDisplay::display(I);

                // 执行 YOLO 检测
                FrameData frame;
                vpImage<vpRGBa> rgb_image;
                vpImageConvert::convert(I, rgb_image);
                frame.image = rgb_image;
                frame.timestamp = 0;

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

                // 保存结果
                if (config.save_results && frame_idx % 10 == 0) {  // 每10帧保存一次
                    saveDetectionResult(I, detection_result, config.output_dir, frame_idx);
                }

                stats.endFrame();
                frame_idx++;

                // 显示图像
                vpDisplay::flush(I);

                // 检查键盘输入
                if (vpDisplay::getKeyboardEvent(I, false)) {
                    char key = vpDisplay::getKeyboardEvent(I);
                    if (key == 'q' || key == 'Q') {
                        running = false;
                    }
                }
            }

        } else if (config.source == "image") {
            std::cout << "加载图像: " << config.input_file << std::endl;
            vpImage<vpRGBa> rgb_image;
            vpImageIo::read(rgb_image, config.input_file);
            vpImageConvert::convert(rgb_image, I);
            display.init(I, 10, 10, "YOLO 检测器测试");

            stats.startFrame();

            // 执行 YOLO 检测
            FrameData frame;
            frame.image = rgb_image;
            frame.timestamp = 0;

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

            stats.endFrame();

            // 保存结果
            if (config.save_results) {
                saveDetectionResult(I, detection_result, config.output_dir, 0);
            }

            // 显示图像
            vpDisplay::display(I);
            vpDisplay::flush(I);
            std::cout << "按任意键退出..." << std::endl;
            vpDisplay::getKeyboardEvent(I);

        } else if (config.source == "video") {
            std::cout << "加载视频: " << config.input_file << std::endl;
            cv::VideoCapture cap(config.input_file);
            if (!cap.isOpened()) {
                std::cerr << "Error: Could not open video file" << std::endl;
                return EXIT_FAILURE;
            }

            int video_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int video_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            I.resize(video_height, video_width);
            display.init(I, 10, 10, "YOLO 检测器测试");

            std::cout << "开始测试... (按 'q' 退出)" << std::endl;
            while (running) {
                cv::Mat frame;
                if (!cap.read(frame)) {
                    break;  // 视频结束
                }

                stats.startFrame();

                // 转换为 ViSP 图像
                vpImage<vpRGBa> rgb_image;
                vpImageConvert::convert(frame, rgb_image);
                vpImageConvert::convert(rgb_image, I);
                vpDisplay::display(I);

                // 执行 YOLO 检测
                FrameData yolo_frame;
                yolo_frame.image = rgb_image;
                yolo_frame.timestamp = 0;

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

                // 保存结果
                if (config.save_results && frame_idx % 10 == 0) {
                    saveDetectionResult(I, detection_result, config.output_dir, frame_idx);
                }

                stats.endFrame();
                frame_idx++;

                // 显示图像
                vpDisplay::flush(I);

                // 检查键盘输入
                if (vpDisplay::getKeyboardEvent(I, false)) {
                    char key = vpDisplay::getKeyboardEvent(I);
                    if (key == 'q' || key == 'Q') {
                        running = false;
                    }
                }
            }

        } else {
            std::cerr << "Error: Invalid source type" << std::endl;
            return EXIT_FAILURE;
        }

        // 显示最终统计
        std::cout << "\n=== 测试完成 ===" << std::endl;
        std::cout << "总帧数: " << stats.getFrameCount() << std::endl;
        std::cout << "平均 FPS: " << std::fixed << std::setprecision(1) << stats.getFPS() << std::endl;
        std::cout << "平均处理时间: " << std::fixed << std::setprecision(1) << stats.getAverageTime() << " ms" << std::endl;
        std::cout << "================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
