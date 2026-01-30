#ifdef HAVE_TENSORRT

#include "TensorRT_detection.h"

// 打印 Cuda 报错
// 函数定义不应放在头文件里，否则当头文件被多个 cpp 调用时，会出现重复定义的问题
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
    if (code != cudaSuccess) {
        const char* err_name = cudaGetErrorName(code);
        const char* err_message = cudaGetErrorString(code);
        printf("runtime error %s: %d  %s failed.\n  code = %s, message = %s", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

// 构造函数
TensorRT_detection::TensorRT_detection(const std::string& file){
    // 1. TensorRT 相关操作
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 1.1 读取 .trt 文件
    _engine_data = load_file(file);
    if (_engine_data.empty()) {
        std::cout << "加载 TRT 引擎文件失败:" << file.c_str();
        return;
    }

    // 1.2 创建运行时，需要日志记录器
    _runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    if (!_runtime) {
        std::cout << "创建 IRuntime 失败";
        return;
    }
    // 1.3 创建推理引擎，需要运行时和序列化 trt 文件，包含反序列化的 .trt 模型数据
    _engine = make_nvshared(_runtime->deserializeCudaEngine(_engine_data.data(), _engine_data.size()));
    if (_engine == nullptr) {
        std::cout << "反序列化 CUDA 引擎失败";
        return;
    }

    // 1.4 创建上下文执行器，需要推理引擎
    _context = make_nvshared(_engine->createExecutionContext());

    // 打印 .trt 模型的输入输出张量的名称和维度，这里与 onnx 中的名称和维度一致
     // for (int i=0, e=_engine->getNbIOTensors(); i<e; i++){
     //     auto const name = _engine->getIOTensorName(i);
     //     auto const size = _engine->getTensorShape(name);
     //     qInfo() << "张量名称: " << name;
     //     for (int j = 0; j < size.nbDims; ++j) {
     //         qInfo() << "维度 " << j << ": " << size.d[j];
     //     }
     //     qInfo() << "";
     // }

    // 2. CUDA 相关操作
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 2.1 创建 CUDA 流，CUDA 流类似于线程，每个任务都必须有一个 CUDA 流，不同的 CUDA 流可以在 GPU 中并行执行任务
    checkRuntime(cudaStreamCreate(&_stream));

    // 2.2 申请 CPU 内存和 GPU 内存，准备好 **_host 和 **_device，分别表示内存中的数据指针和显存中的数据指针
    // 输入内存分配
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));  // CPU 可分页内存
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));    // GPU 内存

    // 输出内存分配（与模型输出维度 [1,7,33600] 匹配）
    checkRuntime(cudaMallocHost(&output_data_host, output_numel * sizeof(float)));  // CPU 内存
    checkRuntime(cudaMalloc(&output_data_device, output_numel * sizeof(float)));    // GPU 内存

    // 3. TensorRT 内存绑定
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    _context->setTensorAddress("images", input_data_device);    // 绑定输入张量
    _context->setTensorAddress("output0", output_data_device);    // 绑定输出张量

    //qInfo() << "TensorRT 检测模块初始化成功！";  // 成功信息

    // 打印输入宽度和高度
    //qInfo() << "初始化前 input_width: " << input_width;
    //qInfo() << "初始化前 input_height: " << input_height;
    // 初始化图像尺寸
    m_input_size = cv::Size(input_width, input_height);
    //qInfo() << "初始化图像尺寸初始化成功！";  // 成功信息
    // 打印初始化后的尺寸
    //qInfo() << "初始化后 m_input_size: " << QString("%1x%2").arg(m_input_size.width).arg(m_input_size.height);

    //4.初始化类别名称
    m_class_names.push_back("luoshuan");  // 类别0：螺栓
    m_class_names.push_back("daoxian");   // 类别1：导线
    m_class_names.push_back("xianjia");   // 类别2：线夹

}

// 析构函数
TensorRT_detection::~TensorRT_detection(){
    // 释放 CPU 内存
    if (input_data_host) checkRuntime(cudaFreeHost(input_data_host));
    if (output_data_host) checkRuntime(cudaFreeHost(output_data_host));

    // 释放 GPU 内存
    if (input_data_device) checkRuntime(cudaFree(input_data_device));
    if (output_data_device) checkRuntime(cudaFree(output_data_device));

    // 释放 CUDA 流
    if (_stream) checkRuntime(cudaStreamDestroy(_stream));

    //qInfo() << "TensorRT 检测模块资源释放成功！";  // 释放成功信息

}

// 读取 .trt 文件
std::vector<unsigned char> TensorRT_detection::load_file(const std::string& file) {  // 返回结果为无符号字符的vector，其数据存储是连成片的
    std::ifstream in(file, std::ios::binary);          // 定义一个数据读取对象，以二进制读取数据
    if (!in.is_open()) return {};                      // 如果没有可读数据则返回空

    in.seekg(0, std::ios::end);                        // seekg函数作用是将指针指向文件终止处
    size_t length = in.tellg();                        // tellg函数作用是返回指针当前位置，此时即为数据长度

    std::vector<uint8_t> data;                              // 定义一个vector用于存储读取数据，仅仅是类头，其数据存储区还是char型data指针
    if (length > 0) {
        in.seekg(0, std::ios::beg);                        // seekg函数作用是将指针指向文件起始处
        data.resize(length);                          // 为vector申请长度为length的数据存储区，默认全部填充 0
        in.read((char*)&data[0], length);             // 为vector的数据存储区读取长度为length的数据
    }
    in.close();                                        // 关闭数据流
    return data;
}


// 图像转换
cv::Mat TensorRT_detection::convertVispToCvMat(const vpImage<vpRGBa>& visp_img)
{
    // 打印输入图像的基本信息
    //qInfo() << "输入图像高度: " << visp_img.getHeight();
    //qInfo() << "输入图像宽度: " << visp_img.getWidth();
    //qInfo() << "输入图像通道数: 4 (vpRGBa 类型)";
    // 使用ViSP格式的图像转换
    cv::Mat cv_img(visp_img.getHeight(), visp_img.getWidth(), CV_8UC4, (void*)visp_img.bitmap);
    // 打印转换后 OpenCV 图像的基本信息
    //qInfo() << "转换后 OpenCV 图像高度: " << cv_img.rows;
    //qInfo() << "转换后 OpenCV 图像宽度: " << cv_img.cols;
    //qInfo() << "转换后 OpenCV 图像通道数: " << cv_img.channels();
    // 检查转换后图像是否为空
    if (cv_img.empty()) {
        std::cout << "转换后的 OpenCV 图像为空！";
    }
    cv::cvtColor(cv_img, cv_img, cv::COLOR_RGBA2RGB);
    // 打印颜色转换后 OpenCV 图像的基本信息
    //qInfo() << "颜色转换后 OpenCV 图像高度: " << cv_img.rows;
    //qInfo() << "颜色转换后 OpenCV 图像宽度: " << cv_img.cols;
    //qInfo() << "颜色转换后 OpenCV 图像通道数: " << cv_img.channels();
    return cv_img;
}

// 图像预处理
cv::Mat TensorRT_detection::preprocessImage(const cv::Mat& image)
{
    //qInfo() << "开始图像预处理";
    //qInfo() << "输入图像尺寸: " << QString("%1x%2").arg(image.size().width).arg(image.size().height);
    //qInfo() << "输入图像通道数: " << image.channels();
    // 确保输入图像是3通道
    cv::Mat input_image;
    if (image.channels() == 1) {
        cv::cvtColor(image, input_image, cv::COLOR_GRAY2RGB);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, input_image, cv::COLOR_RGBA2RGB);
    } else {
        input_image = image.clone();
    }

    // 检查输入图像尺寸是否有效
    if (input_image.empty() || input_image.cols == 0 || input_image.rows == 0) {
        std::cout << "输入图像尺寸无效";
        return cv::Mat();
    }

    //qInfo() << "转换后图像尺寸: " << QString("%1x%2").arg(input_image.size().width).arg(input_image.size().height);
    //qInfo() << "转换后图像通道数: " << input_image.channels();
    // 打印目标尺寸信息
    //qInfo() << "目标尺寸 (m_input_size): " << QString("%1x%2").arg(m_input_size.width).arg(m_input_size.height);

    // 检查 m_input_size 是否有效
    if (m_input_size.width == 0 || m_input_size.height == 0) {
        std::cout << "目标尺寸无效";
        return cv::Mat();
    }
    // 调整大小
    cv::Mat resized;
    cv::resize(input_image, resized, m_input_size);

    // 转换为浮点数并归一化
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 确保正确的通道顺序 (HWC -> CHW 如果需要)
    // 检查模型需要的输入格式
    // 如果是 CHW 格式，需要转换
    cv::Mat chw_img;
    cv::dnn::blobFromImage(float_img, chw_img, 1.0, cv::Size(), cv::Scalar(), true, false);

    return chw_img;
}

// 模型推理
void TensorRT_detection::infer_trtmodel(FrameData &frame, DetectionResult &result){
    // 记录起始时间
    auto start = std::chrono::high_resolution_clock::now();

    // 转换图像格式
    cv::Mat cv_img = convertVispToCvMat(frame.image);
    //qInfo() << "图像格式转换成功";
    // 图像预处理
    cv::Mat preprocessed = preprocessImage(cv_img);
    //qInfo() << "图像预处理成功";

    // 记录起始时间
    //auto start = std::chrono::high_resolution_clock::now();

    //将预处理结果复制到输入缓冲区
    std::memcpy(input_data_host, preprocessed.ptr<float>(), input_numel * sizeof(float));
    //qInfo() << "预处理结果复制到输入缓冲区成功";

    // 将输入图片从 CPU 内存拷贝至 GPU 内存
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel *sizeof(float), cudaMemcpyHostToDevice, _stream));
    // 模型推理
    bool success = _context->enqueueV3(_stream);
    //qInfo() << "模型推理成功";

    // 将输出结果从 GPU 内存拷贝至 CPU 内存
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, output_numel * sizeof(float), cudaMemcpyDeviceToHost, _stream));
    // 等待直到 _stream 流的工作完成，在这行之前不要做与输出结果处理或展示相关的操作
    checkRuntime(cudaStreamSynchronize(_stream));


    // 记录起始时间
    //auto start = std::chrono::high_resolution_clock::now();
    // 后处理

    result.timestamp = 0;
    result.success = false;

    try {
        // 获取输出张量信息
        std::vector<int64_t> shape = {output_batch, output_dim1, output_dim2};

        // 验证输出维度
        if (shape.size() != 3 || shape[1] != 7) {
            std::cout << "Unexpected output shape: ["
                       << shape[0] << ", " << shape[1] << ", " << shape[2] << "]";
            return;
        }
        const int box_classconfidence = static_cast<int>(shape[1]); // 4+类别数
        const int num_predictions = static_cast<int>(shape[2]);     // 预测点数量
        const float* data = output_data_host;

        // 解析检测结果
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> class_ids;

        for (int64_t i = 0; i < num_predictions; ++i) {
            // 解析边界框坐标
            float x = data[0 * num_predictions + i];
            float y = data[1 * num_predictions + i];
            float w = data[2 * num_predictions + i];
            float h = data[3 * num_predictions + i];

            float x1 = x - w / 2;
            float y1 = y - h / 2;
            float x2 = x + w / 2;
            float y2 = y + h / 2;

            // 提取类别概率并找出最大置信度
            int class_id = -1;
            float max_conf = 0.0f;
            for (int c = 0; c < box_classconfidence - 4; ++c) {
                float conf = data[(4 + c) * num_predictions + i];  // 跳过前4个坐标值
                if (conf > max_conf) {
                    max_conf = conf;
                    class_id = c;
                }
            }

            // 跳过低置信度检测
            if (max_conf < m_confidence_threshold) continue;

            // 转换为图像坐标
            float scale_x = cv_img.cols / static_cast<float>(m_input_size.width);
            float scale_y = cv_img.rows / static_cast<float>(m_input_size.height);

            int left = static_cast<int>(x1 * scale_x);
            int top = static_cast<int>(y1 * scale_y);
            int width = static_cast<int>((x2 - x1) * scale_x);
            int height = static_cast<int>((y2 - y1) * scale_y);

            // 确保在图像范围内
            left = std::max(0, std::min(left, cv_img.cols - 1));
            top = std::max(0, std::min(top, cv_img.rows - 1));
            width = std::max(1, std::min(width, cv_img.cols - left));
            height = std::max(1, std::min(height, cv_img.rows - top));

            // 检查边界框尺寸是否合理
            if (width > 0 && height > 0) {
                boxes.emplace_back(left, top, width, height);
                confidences.push_back(max_conf);
                class_ids.push_back(class_id);
            }
        }

        // 应用NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, m_confidence_threshold, m_nms_threshold, indices);

        int bolt_count = 0;
        // 构建最终结果
        for (int idx : indices) {
            DetectionResult::BoltDetection bolt;
            bolt.bounding_box = vpRect(boxes[idx].x, boxes[idx].y, boxes[idx].width, boxes[idx].height);
            bolt.confidence = confidences[idx];
            bolt.class_id = class_ids[idx];

            if (bolt.class_id < static_cast<int>(m_class_names.size())) {
                bolt.class_name = m_class_names[bolt.class_id];
            } else {
                bolt.class_name = "unknown";
            }
            bolt_count++;
            // qDebug().nospace() << "螺栓 #" << bolt_count
            //                    << ": 位置=[" << boxes[idx].x << ", " << boxes[idx].y
            //                    << "], 尺寸=[" << boxes[idx].width << "x" << boxes[idx].height
            //                    << "], 置信度=" << QString::number(confidences[idx], 'f', 3)
            //                    << ", 类别=" << bolt.class_id << "(" << bolt.class_name << ")";

            result.bolts.push_back(bolt);
        }

        result.success = true;
        result.message = "Detection successful";

    } catch (const std::exception& e) {
        result.message = "Error: " + std::string(e.what());
        std::cout << "Detection error:" << e.what();
    }

    auto end = std::chrono::high_resolution_clock::now();
    result.processing_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double fps = 1000.0 / result.processing_time_ms;

    //qDebug() << "检测到" << result.bolts.size() << "个螺栓";
    //qDebug() << "处理时间: " << result.processing_time_ms << " ms";
    //qDebug() << "检测帧率: " << fps << " FPS";

    //qInfo() << "推理完成";
}

#endif
