#ifndef APPCONFIG_H
#define APPCONFIG_H

#include <string>
#include <vector>

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpCameraParameters.h>
#include <visp3/core/vpImage.h>

// Eigen 库头文件
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

// 应用配置结构体
struct AppConfig {
    // 机器人参数
    std::string robot_ip = "192.168.31.100";  // UR12e机器人IP
    
    // AprilTag参数
    double tag_size = 0.03;                    // AprilTag标签尺寸, 单位米
    int tag_quad_decimate = 2;                 // AprilTag检测时的图像降采样因子
    bool display_tag = true;                   // 是否在图像上显示检测到的标签
    
    // 相机参数
    unsigned int camera_width = 1280;          // 相机宽度
    unsigned int camera_height = 720;          // 相机高度
    
    // 控制参数
    double convergence_threshold = 0.00005;    // 收敛阈值
    double desired_loop_time_ms = 20.0;        // 期望循环时间
    double force_z_threshold = -15.0;          // 力传感器阈值
    
    // 遥操作参数
    double fine_linear_step = 0.005;           // 平移精细步长
    double coarse_linear_step = 0.015;         // 平移快速步长
    double fine_angular_step = 0.025;          // 旋转精细步长
    double coarse_angular_step = 0.120;        // 旋转快速步长
    double fine_joint_step = 0.005;            // 关节精细步长
    double coarse_joint_step = 0.010;          // 关节快速步长
    
    // 外参矩阵
    vpColVector ePc = vpColVector{0.141335, 0.174283, 0.0915465, -0.556397, 1.79421, -1.77789};       // 相机外参初始值
    
    // 显示参数
    bool verbose = false;                      // 是否输出详细调试信息
    bool plot = false;                         // 是否实时绘制曲线图
    bool adaptive_gain = false;                // 是否使用自适应增益
    bool task_sequencing = false;              // 是否使用任务序列化
    
    // 安全位姿（关节角度，单位弧度）
    vpColVector safe_joint_position = vpColVector(6, 0);
    
    AppConfig() {
        // 初始化安全位姿
        safe_joint_position[0] =  vpMath::rad(152.65);
        safe_joint_position[1] = -vpMath::rad(110.89);
        safe_joint_position[2] =  vpMath::rad(119.46);
        safe_joint_position[3] = -vpMath::rad(103.64);
        safe_joint_position[4] =  vpMath::rad(90.38);
        safe_joint_position[5] = -vpMath::rad(107.6);
    }
};

// 辅助函数：将ViSP深度图像转换为OpenCV矩阵
inline cv::Mat vispDepthToCvMat(const vpImage<float> &depth) {
    cv::Mat mat(depth.getHeight(), depth.getWidth(), CV_32F);
    for (unsigned int i = 0; i < depth.getHeight(); i++) {
        for (unsigned int j = 0; j < depth.getWidth(); j++) {
            mat.at<float>(i, j) = depth[i][j];
        }
    }
    return mat;
}

// 辅助函数：将Eigen矩阵转换为ViSP矩阵
inline vpHomogeneousMatrix eigenToVispMatrix(const Eigen::Matrix4d &eigen_mat) {
    vpHomogeneousMatrix visp_mat;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            visp_mat[i][j] = eigen_mat(i, j);
        }
    }
    return visp_mat;
}

#endif // APPCONFIG_H
