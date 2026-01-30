#!/bin/bash

# YOLO 检测器测试脚本

echo "=== YOLO 检测器测试脚本 ==="
echo ""
echo "测试程序位置: build/test/test_yolo_detector"
echo ""
echo "使用说明:"
echo "1. 相机测试: ./run_test.sh camera <模型路径>"
echo "2. 图像测试: ./run_test.sh image <模型路径> <图像路径>"
echo "3. 视频测试: ./run_test.sh video <模型路径> <视频路径>"
echo ""
echo "示例:"
echo "   ./run_test.sh camera ../models/yolo_model.trt"
echo "   ./run_test.sh image ../models/yolo_model.trt ../test_data/test_image.jpg"
echo "   ./run_test.sh video ../models/yolo_model.trt ../test_data/test_video.mp4"
echo ""

if [ $# -lt 2 ]; then
    echo "错误: 参数不足"
    echo "使用方法: ./run_test.sh <测试类型> <模型路径> [输入文件路径]"
    exit 1
fi

TEST_TYPE=$1
MODEL_PATH=$2
INPUT_FILE=""

if [ $# -ge 3 ]; then
    INPUT_FILE=$3
fi

TEST_EXECUTABLE="../build/test/test_yolo_detector"

if [ ! -f "$TEST_EXECUTABLE" ]; then
    echo "错误: 测试程序未找到，请先构建"
    echo "构建命令: cd ../build/test && cmake ../../test && make"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "警告: 模型文件未找到: $MODEL_PATH"
    echo "将使用替代实现进行测试"
fi

echo "开始测试..."
echo "测试类型: $TEST_TYPE"
echo "模型路径: $MODEL_PATH"
if [ ! -z "$INPUT_FILE" ]; then
    echo "输入文件: $INPUT_FILE"
fi
echo ""

# 运行测试
if [ "$TEST_TYPE" == "camera" ]; then
    $TEST_EXECUTABLE --model "$MODEL_PATH" --source camera --save --stats
elif [ "$TEST_TYPE" == "image" ]; then
    if [ -z "$INPUT_FILE" ]; then
        echo "错误: 图像测试需要指定图像路径"
        exit 1
    fi
    $TEST_EXECUTABLE --model "$MODEL_PATH" --source image --input "$INPUT_FILE" --save --stats
elif [ "$TEST_TYPE" == "video" ]; then
    if [ -z "$INPUT_FILE" ]; then
        echo "错误: 视频测试需要指定视频路径"
        exit 1
    fi
    $TEST_EXECUTABLE --model "$MODEL_PATH" --source video --input "$INPUT_FILE" --save --stats
else
    echo "错误: 无效的测试类型: $TEST_TYPE"
    echo "支持的测试类型: camera, image, video"
    exit 1
fi

echo ""
echo "测试完成！"
echo "结果保存在: results/ 目录"
echo ""
