#ifndef TENSORRT_DETECTION_H
#define TENSORRT_DETECTION_H

/**
 * @file TensorRT_detection.h
 * @brief TensorRT 目标检测模块头文件
 * @details 该文件定义了基于 TensorRT 实现的 YOLO 目标检测模块，用于在 IBVS 遥操作系统中进行实时目标检测。
 */

// 先取消可能的冲突宏
#undef MAJOR_VERSION
#undef MINOR_VERSION
#undef PATCH_LEVE

#include <string>
#include <vector>
#include <memory>
#include <fstream>

// 条件包含 TensorRT 相关头文件
#ifdef HAVE_TENSORRT
// TensorRT 相关头文件
#include <NvInfer.h>
#include <NvInferRuntime.h>
// CUDA 相关头文件
#include <cuda_runtime.h>
#endif

//图像处理
#include <visp3/core/vpImage.h>
#include <visp3/core/vpRect.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/core/vpIoTools.h>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/imgproc.hpp>

/**
 * @struct FrameData
 * @brief 帧数据结构
 * @details 存储单帧图像数据及其时间戳，用于在系统各模块间传递图像信息。
 */
struct FrameData {
    vpImage<vpRGBa> image;        // ViSP图像格式，包含RGBa通道数据
    uint64_t timestamp;           // 时间戳，用于同步和记录
};

/**
 * @struct DetectionResult
 * @brief 检测结果结构
 * @details 存储目标检测的结果信息，包括检测到的目标、处理时间等。
 */
struct DetectionResult {
    /**
     * @struct BoltDetection
     * @brief 螺栓检测结果结构
     * @details 存储单个螺栓的检测信息，包括边界框、置信度、类别等。
     */
    struct BoltDetection {
        vpRect bounding_box;      // 螺栓边界框，包含位置和尺寸信息
        float confidence;         // 检测置信度，范围0-1
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
/**
 * @macro checkRuntime
 * @brief 检查 CUDA 运行时错误的宏
 * @param op CUDA 操作
 * @details 调用 __check_cuda_runtime 函数检查 CUDA 操作是否成功，失败则打印错误信息
 */
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)

// 条件编译 TensorRT 相关代码
#ifdef HAVE_TENSORRT

/**
 * @brief 严重性级别字符串转换函数
 * @param t TensorRT 日志严重性级别
 * @return 对应的字符串表示
 * @details 将 TensorRT 的日志严重性级别转换为可读的字符串
 */
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

/**
 * @brief 创建 TensorRT 对象的智能指针
 * @tparam _T TensorRT 对象类型
 * @param ptr 原始指针
 * @return 智能指针
 * @details 模板函数，用于创建管理 TensorRT 对象生命周期的智能指针
 */
template<typename _T>
std::shared_ptr<_T> make_nvshared(_T *ptr) {
    return std::shared_ptr<_T>(ptr);
}

/**
 * @class TRTLogger
 * @brief TensorRT 日志记录器类
 * @details 继承自 nvinfer1::ILogger，用于处理 TensorRT 的日志输出
 */
class TRTLogger : public nvinfer1::ILogger {
public:
    /**
     * @brief 日志记录方法
     * @param severity 日志严重性级别
     * @param msg 日志消息
     * @details 重写父类方法，根据严重性级别输出日志信息
     */
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

/**
 * @class TensorRT_detection
 * @brief TensorRT 目标检测类
 * @details 基于 TensorRT 实现的 YOLO 目标检测模块，用于实时检测图像中的目标
 */
class TensorRT_detection {
public:
    /**
     * @brief 默认构造函数
     * @details 创建未初始化的 TensorRT 检测对象
     */
    TensorRT_detection() = default;
    
    /**
     * @brief 构造函数
     * @param file TensorRT 引擎文件路径
     * @details 初始化 TensorRT 检测对象，加载模型文件并设置相关参数
     */
    TensorRT_detection(const std::string& file);
    
    /**
     * @brief 析构函数
     * @details 释放 TensorRT 检测对象占用的资源
     */
    ~TensorRT_detection();
    
    /**
     * @brief 模型推理方法
     * @param frame 输入帧数据
     * @param result 输出检测结果
     * @details 使用 TensorRT 引擎对输入图像进行目标检测，并返回检测结果
     */
    void infer_trtmodel(FrameData &frame, DetectionResult &result);
    
    /**
     * @brief 图像转换方法
     * @param visp_img ViSP 格式图像
     * @return OpenCV 格式图像
     * @details 将 ViSP 格式的图像转换为 OpenCV 格式，便于后续处理
     */
    cv::Mat convertVispToCvMat(const vpImage<vpRGBa>& visp_img);
    
    /**
     * @brief 图像预处理方法
     * @param image OpenCV 格式图像
     * @return 预处理后的图像
     * @details 对输入图像进行预处理，包括调整大小、归一化等操作
     */
    cv::Mat preprocessImage(const cv::Mat& image);
    
    /**
     * @brief 检查初始化状态
     * @return 初始化是否成功
     * @details 检查 TensorRT 检测对象是否成功初始化
     */
    bool isInitialized() const { return _initialized; }
    
private:
    /**
     * @brief 读取 .trt 文件
     * @param file 文件路径
     * @return 文件二进制数据
     * @details 读取 TensorRT 引擎文件的二进制数据
     */
    std::vector<unsigned char> load_file(const std::string& file);
    
    float m_confidence_threshold = 0.1f;  // 置信度阈值，低于此值的检测结果会被过滤
    float m_nms_threshold = 0.4f;         // NMS阈值，用于去除重叠的检测结果
    std::vector<std::string> m_class_names;  // 类别名称列表
    bool _initialized = false;             // 初始化成功标志
    
public:
    // 输出结果

private:
    // 图像尺寸
    cv::Size m_input_size;  // 模型输入图像尺寸
    
    // TensorRT 推理用的工具
    std::vector<unsigned char> _engine_data;                           // 记录 .trt 模型的二进制序列化格式数据
    TRTLogger logger;                                                  // 打印 TensorRT 的错误信息
    std::shared_ptr<nvinfer1::IRuntime> _runtime = nullptr;            // 运行时，即推理引擎的支持库和函数等
    std::shared_ptr<nvinfer1::ICudaEngine> _engine = nullptr;          // 推理引擎，包含反序列化的 .trt 模型数据
    std::shared_ptr<nvinfer1::IExecutionContext> _context = nullptr;   // 上下文执行器，用于做模型推理
    cudaStream_t _stream = nullptr;  // CUDA 流，用于并行处理

    // 定义模型输入输出尺寸
    int input_batch = 1;     // 批处理大小
    int input_channel = 3;   // 输入通道数
    int input_height = 1280; // 输入高度
    int input_width = 1280;  // 输入宽度
    int output_batch = 1;    // 与输入 batch 一致
    int output_dim1 = 7;     // 输出的第二维（如类别数或特征维度）
    int output_dim2 = 33600; // 输出的第三维（特征数量或预测结果）

    // 准备好 **_host 和 **_device，分别表示内存中的数据指针和显存中的数据指针
    // input 数据（与输入维度匹配）
    int input_numel = input_batch * input_channel * input_height * input_width;  // 输入元素总数
    float* input_data_host = nullptr;    // 主机内存（CPU）中的输入数据
    float* input_data_device = nullptr;  // 设备内存（GPU）中的输入数据

    // output 数据（与模型输出维度 [1,7,33600] 匹配）
    // 总元素数量 = batch × 维度1 × 维度2
    int output_numel = output_batch * output_dim1 * output_dim2;  // 输出元素总数
    float* output_data_host = nullptr;   // 主机内存（CPU）中的输出结果
    float* output_data_device = nullptr; // 设备内存（GPU）中的输出结果

};

#else

/**
 * @class TensorRT_detection
 * @brief 当没有 TensorRT 时的替代实现
 * @details 提供与 TensorRT 版本相同的接口，但不执行实际的检测操作
 */
class TensorRT_detection {
public:
    /**
     * @brief 默认构造函数
     * @details 创建未初始化的检测对象
     */
    TensorRT_detection() = default;
    
    /**
     * @brief 构造函数
     * @param file 模型文件路径
     * @details 打印 TensorRT 未找到的信息
     */
    TensorRT_detection(const std::string& file) {
        std::cout << "TensorRT 未找到，YOLO 检测不可用" << std::endl;
    }
    
    /**
     * @brief 析构函数
     * @details 空实现
     */
    ~TensorRT_detection() {}
    
    /**
     * @brief 模型推理方法（替代实现）
     * @param frame 输入帧数据
     * @param result 输出检测结果
     * @details 设置检测结果为失败，并说明原因
     */
    void infer_trtmodel(FrameData &frame, DetectionResult &result) {
        result.success = false;
        result.message = "TensorRT 未找到，YOLO 检测不可用";
    }
    
    /**
     * @brief 图像转换方法（替代实现）
     * @param visp_img ViSP 格式图像
     * @return 空的 OpenCV 图像
     * @details 返回空图像
     */
    cv::Mat convertVispToCvMat(const vpImage<vpRGBa>& visp_img) {
        return cv::Mat();
    }
    
    /**
     * @brief 图像预处理方法（替代实现）
     * @param image OpenCV 格式图像
     * @return 空的 OpenCV 图像
     * @details 返回空图像
     */
    cv::Mat preprocessImage(const cv::Mat& image) {
        return cv::Mat();
    }
};

#endif

#endif // TENSORRT_DETECTION_H
