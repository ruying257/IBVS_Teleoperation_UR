/**
 * @file SystemController.h
 * @brief 系统控制器类，负责协调机器人和传感器的操作
 * 
 * 该类实现了一个状态机，用于管理机器人的不同操作模式（如视觉伺服、遥操作等）。
 * 它整合了机器人模型、传感器（如相机和力传感器）、检测算法（如AprilTag检测）
 * 以及控制任务（如视觉伺服）。
 */
#ifndef SYSTEMCONTROLLER_H
#define SYSTEMCONTROLLER_H

#include <memory>
#include <vector>

#include <visp3/core/vpConfig.h>
#include <visp3/core/vpImage.h>
#include <visp3/core/vpColVector.h>
#include <visp3/core/vpHomogeneousMatrix.h>
#include <visp3/core/vpPoint.h>
#include <visp3/robot/vpRobotUniversalRobots.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/vs/vpServo.h>
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/gui/vpPlot.h>

#include "AppConfig.h"
#include "RobotTeleoperation.h"

class SystemController {
public:
    // 状态机枚举
    enum State {
        STATE_IBVS,         // 视觉伺服状态
        STATE_WAIT_SELECT,  // 等待选择状态
        STATE_APPROACH,     // 接近目标状态
        STATE_TELEOP        // 遥操作状态
    };

    SystemController(const AppConfig& config);
    ~SystemController();

    bool initialize();
    void run();
    void stop();

private:
    // 配置参数
    AppConfig config;
    
    // 核心组件
    vpRobotUniversalRobots robot;
    vpRealSense2 rs;
    vpDetectorAprilTag detector;
    vpServo task;
    
    // 遥操作
    RobotTeleoperation teleop;
    
    // 显示相关
#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
    std::shared_ptr<vpDisplay> display;
#else
    vpDisplay *display;
#endif
    vpImage<unsigned char> I;
    vpPlot *plotter = nullptr;
    
    // 特征点和目标点
    std::vector<vpFeaturePoint> p, pd;
    std::vector<vpPoint> point;
    std::vector<vpImagePoint> *traj_corners = nullptr;
    
    // 变换矩阵
    vpHomogeneousMatrix eMc, cMo, oMo, cdMo;
    
    // 相机参数
    vpCameraParameters cam;
    
    // 速度向量
    vpColVector velocity_visual_servo;
    vpColVector velocity_send;
    vpColVector velocity_zero;
    vpColVector force_torque;
    
    // 控制变量
    State current_state;
    bool final_quit;
    bool send_velocities;
    bool servo_started;
    bool freeze_translation;
    int converged_cont;
    int iter_plot;
    double force_z;
    
    // 方法
    void process_ibvs();
    void process_teleop();
    void display_point_trajectory(const std::vector<vpImagePoint> &vip);
    void handle_events();
    void calculate_loop_time();
};

#endif // SYSTEMCONTROLLER_H
