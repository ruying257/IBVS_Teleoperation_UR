#ifndef TENSORRT_DETECTION_H
#define TENSORRT_DETECTION_H

// 先取消可能的冲突宏
#undef MAJOR_VERSION
#undef MINOR_VERSION
#undef PATCH_LEVE

#include <string>
#include <vector>
#include <memory>
#include <fstream>
// TensorRT 相关头文件
#include <NvInfer.h>
#include <NvInferRuntime.h>
// CUDA 相关头文件
#include <cuda_runtime.h>

//图像处理
#include <visp3/core/vpImage.h>
#include <visp3/core/vpRect.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpIoTools.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>

struct FrameData {
    vpImage<vpRGBa> image;        // ViSP图像格式
    uint64_t timestamp;           // 时间戳
};

struct DetectionResult {
    struct BoltDetection {
        vpRect bounding_box;      // 螺栓边界框
        float confidence;         // 检测置信度
        int class_id;             // 类别ID（默认为0）
        std::string class_name;   // 类别名称
    };

    std::vector<BoltDetection> bolts;  // 检测到的螺栓列表
    uint64_t processing_time_ms;       // 处理耗时(ms)
    uint64_t timestamp;                // 时间戳
    bool success;                      // 处理是否成功
    std::string message;               // 附加信息（错误消息等）
};

// 用于打印 CUDA 报错 - 宏定义
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// 用于打印 TensorRT 报错，准备一个 logger 类，打印构建 TensorRT 推理模型过程中的一些错误或警告，按照指定的严重性程度 (severity) 打印信息
    // 内联函数可以放在头文件，因为内联函数不会产生独立的符号，不会引起多重定义的问题‌
    inline const char* severity_string(nvinfer1::ILogger::Severity t) {
    switch (t) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR: return "error";
    case nvinfer1::ILogger::Severity::kWARNING: return "warning";
    case nvinfer1::ILogger::Severity::kINFO: return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
    default: return "unknown";
    }
}

    // 定义一个 share_ptr 的智能指针
    // 模板的定义必须放在头文件中，因为模板的实例化需要在编译时进行
    // 这里定义了一个模板函数 make_nvshared，用于创建一个 shared_ptr 智能指针，管理传入的原始指针 ptr 的生命周期
    template<typename _T>
    std::shared_ptr<_T> make_nvshared(_T *ptr) {
        return std::shared_ptr<_T>(ptr);
    }

class TRTLogger : public nvinfer1::ILogger {
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            if (severity == Severity::kWARNING)
                std::cout << severity_string(severity) << ": " << msg;
            else if (severity == Severity::kERROR)
                std::cout << severity_string(severity) << ": " << msg;
            else
                std::cout << severity_string(severity) << ": " << msg;
        }
    }
};
//
class TensorRT_detection {
public:
    TensorRT_detection() = default;
    TensorRT_detection(const std::string& file);
    ~TensorRT_detection();
    // 模型推理
    void infer_trtmodel(FrameData &frame, DetectionResult &result);
    // 图像转换
    cv::Mat convertVispToCvMat(const vpImage<vpRGBa>& visp_img);
    // 预处理
    cv::Mat preprocessImage(const cv::Mat& image);
private:
    // 读取 .trt 文件
    std::vector<unsigned char> load_file(const std::string& file);
    float m_confidence_threshold = 0.1f;  // 置信度阈值
    float m_nms_threshold = 0.4f;         // NMS阈值
    std::vector<std::string> m_class_names;  // 添加类别名称列表
public:
    // 输出结果

private:
    // 图像尺寸
    cv::Size m_input_size;
    // TensorRT 推理用的工具
    std::vector<unsigned char> _engine_data;                           // 记录 .trt 模型的二进制序列化格式数据
    TRTLogger logger;                                                  // 打印 TensorRT 的错误信息
    std::shared_ptr<nvinfer1::IRuntime> _runtime = nullptr;            // 运行时，即推理引擎的支持库和函数等
    std::shared_ptr<nvinfer1::ICudaEngine> _engine = nullptr;          // 推理引擎，包含反序列化的 .trt 模型数据
    std::shared_ptr<nvinfer1::IExecutionContext> _context = nullptr;   // 上下文执行器，用于做模型推理
    cudaStream_t _stream = nullptr;

    // 定义模型输入输出尺寸
    // int input_batch = 1;
    // int input_channel = 3;
    // int input_height = 720;
    // int input_width = 1280;
    // int output_batch = 1;          // 与输入 batch 一致
    // int output_dim1 = 7;           // 输出的第二维（如类别数或特征维度）
    // int output_dim2 = 33600;       // 输出的第三维（特征数量或预测结果）

    int input_batch = 1;
    int input_channel = 3;
    int input_height = 1280;
    int input_width = 1280;
    int output_batch = 1;          // 与输入 batch 一致
    int output_dim1 = 7;           // 输出的第二维（如类别数或特征维度）
    int output_dim2 = 33600;       // 输出的第三维（特征数量或预测结果）

    // 准备好 **_host 和 **_device，分别表示内存中的数据指针和显存中的数据指针
    // input 数据（与输入维度匹配）
    int input_numel = input_batch * input_channel * input_height * input_width;
    float* input_data_host = nullptr;    // 主机内存（CPU）中的输入数据
    float* input_data_device = nullptr;  // 设备内存（GPU）中的输入数据

    // output 数据（与模型输出维度 [1, 7, 33600] 匹配）
    // 总元素数量 = batch × 维度1 × 维度2
    int output_numel = output_batch * output_dim1 * output_dim2;
    float* output_data_host = nullptr;   // 主机内存（CPU）中的输出结果
    float* output_data_device = nullptr; // 设备内存（GPU）中的输出结果

};

#endif // TENSORRT_DETECTION_H
