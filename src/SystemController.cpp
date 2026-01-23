#include "SystemController.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

#include <visp3/core/vpMath.h>
#include <visp3/core/vpMeterPixelConversion.h>
#include <visp3/gui/vpDisplayFactory.h>
#include <visp3/gui/vpServoDisplay.h>

SystemController::SystemController(const AppConfig& config) 
    : config(config),
      current_state(STATE_IBVS),
      final_quit(false),
      send_velocities(false),
      servo_started(false),
      freeze_translation(false),
      converged_cont(0),
      iter_plot(0),
      force_z(0),
      velocity_visual_servo(6, 0),
      velocity_send(6, 0),
      velocity_zero(6, 0),
      force_torque(6, 0),
      I(config.camera_height, config.camera_width) {
}

SystemController::~SystemController() {
    stop();
}

void SystemController::stop() {
    if (teleop.isRunning()) {
        teleop.stop();
    }
    
    if (plotter != nullptr) {
        delete plotter;
        plotter = nullptr;
    }
    
    if (traj_corners) {
        delete[] traj_corners;
        traj_corners = nullptr;
    }
    
#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
    if (display != nullptr) {
        delete display;
        display = nullptr;
    }
#endif
}

bool SystemController::initialize() {
    try {
        // 连接机器人
        std::cout << "连接到机器人: " << config.robot_ip << std::endl;
        robot.connect(config.robot_ip);
        
        // 警告用户
        std::cout << "WARNING: This example will move the robot! "
                  << "Please make sure to have the user stop button at hand!" << std::endl
                  << "Press Enter to continue..." << std::endl;
        std::cin.ignore();
        
        // 移动到安全位姿
        std::cout << "Move to joint position: " << config.safe_joint_position.t() << std::endl;
        robot.setRobotState(vpRobot::STATE_POSITION_CONTROL);
        robot.setPosition(vpRobot::JOINT_STATE, config.safe_joint_position);
        
        // 初始化相机
        std::cout << "初始化相机..." << std::endl;
        rs2::config rs_config;
        rs_config.enable_stream(RS2_STREAM_COLOR, config.camera_width, config.camera_height, RS2_FORMAT_RGBA8, 30);
        rs_config.enable_stream(RS2_STREAM_DEPTH, config.camera_width, config.camera_height, RS2_FORMAT_Z16, 30);
        rs_config.disable_stream(RS2_STREAM_INFRARED);  // D405没有红外流
        rs.open(rs_config);
        
        // 设置相机外参
        ePc = config.ePc;
        eMc.buildFrom(ePc);
        std::cout << "eMc:\n" << eMc << "\n";
        
        // 获取相机内参
        cam = rs.getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithDistortion);
        std::cout << "cam:\n" << cam << "\n";
        
        // 创建显示
#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
        display = vpDisplayFactory::createDisplay(I, 10, 10, "Color image");
#else
        display = vpDisplayFactory::allocateDisplay(I, 10, 10, "Color image");
#endif
        
        // 初始化AprilTag检测器
        vpDetectorAprilTag::vpAprilTagFamily tagFamily = vpDetectorAprilTag::TAG_36h11;
        vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod = vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS;
        detector.setAprilTagFamily(tagFamily);
        detector.setAprilTagPoseEstimationMethod(poseEstimationMethod);
        detector.setDisplayTag(config.display_tag);
        detector.setAprilTagQuadDecimate(config.tag_quad_decimate);
        
        // 初始化目标变换矩阵
        cdMo.buildFrom(vpTranslationVector(0, 0, config.tag_size * 10),
                      vpRotationMatrix({ 1, 0, 0, 0, -1, 0, 0, 0, -1 }));
        
        // 初始化特征点
        point.resize(4);
        point[0].setWorldCoordinates(-config.tag_size / 2., -config.tag_size / 2., 0);
        point[1].setWorldCoordinates( config.tag_size / 2., -config.tag_size / 2., 0);
        point[2].setWorldCoordinates( config.tag_size / 2.,  config.tag_size / 2., 0);
        point[3].setWorldCoordinates(-config.tag_size / 2.,  config.tag_size / 2., 0);
        
        p.resize(4);
        pd.resize(4);
        
        // 初始化视觉伺服任务
        task.setServo(vpServo::EYEINHAND_CAMERA);
        task.setInteractionMatrixType(vpServo::CURRENT);
        
        if (config.adaptive_gain) {
            vpAdaptiveGain lambda(1.5, 0.4, 30);
            task.setLambda(lambda);
        } else {
            task.setLambda(0.5);
        }
        
        for (size_t i = 0; i < p.size(); i++) {
            task.addFeature(p[i], pd[i]);
        }
        
        // 初始化绘图
        if (config.plot) {
            plotter = new vpPlot(
                2, static_cast<int>(250 * 2), 500,
                static_cast<int>(I.getWidth()) + 80, 10,
                "Real time curves plotter"
            );
            plotter->setTitle(0, "Visual features error");
            plotter->setTitle(1, "Camera velocities");
            plotter->initGraph(0, 8);
            plotter->initGraph(1, 6);
            plotter->setLegend(0, 0, "error_feat_p1_x");
            plotter->setLegend(0, 1, "error_feat_p1_y");
            plotter->setLegend(0, 2, "error_feat_p2_x");
            plotter->setLegend(0, 3, "error_feat_p2_y");
            plotter->setLegend(0, 4, "error_feat_p3_x");
            plotter->setLegend(0, 5, "error_feat_p3_y");
            plotter->setLegend(0, 6, "error_feat_p4_x");
            plotter->setLegend(0, 7, "error_feat_p4_y");
            plotter->setLegend(1, 0, "vc_x");
            plotter->setLegend(1, 1, "vc_y");
            plotter->setLegend(1, 2, "vc_z");
            plotter->setLegend(1, 3, "wc_x");
            plotter->setLegend(1, 4, "wc_y");
            plotter->setLegend(1, 5, "wc_z");
        }
        
        // 初始化力传感器
        robot.zeroFTSensor();
        robot.getForceTorque(vpRobot::CAMERA_FRAME, force_torque);
        force_z = force_torque[2];
        
        // 启动遥操作
        RobotTeleoperation::printInstructions();
        teleop.setStepSizes(
            config.fine_linear_step, config.coarse_linear_step,
            config.fine_angular_step, config.coarse_angular_step,
            config.fine_joint_step, config.coarse_joint_step
        );
        
        if (!teleop.start()) {
            std::cerr << "无法启动键盘监听" << std::endl;
            return false;
        }
        
        std::cout << "\n键盘遥操作已启动:收敛前为 IBVS 全 6 维控制；"
                  << "收敛后:键盘控制 XYZ, 取消除rz以外的旋转。\n"
                  << "按 Q 退出, 空格急停, 鼠标左键开/关速度发送, 右键退出程序。\n";
        
        // 设置机器人状态
        robot.set_eMc(eMc);
        robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);
        
        return true;
        
    } catch (const std::exception &e) {
        std::cout << "初始化失败: " << e.what() << std::endl;
        return false;
    }
}

void SystemController::run() {
    static double t_init_servo = vpTime::measureTimeMs();
    
    while (!final_quit) {
        double t_start = vpTime::measureTimeMs();
        
        // 从相机获取并显示图像
        rs.acquire(I);
        vpDisplay::display(I);
        
        // 读取键盘控制
        RobotTeleoperation::ControlVector teleop_control = teleop.getControlVector();
        if (teleop_control.exit_requested) {
            std::cout << "\n收到键盘退出请求, 准备退出程序" << std::endl;
            final_quit = true;
        }
        if (teleop_control.is_estop) {
            std::cout << "[键盘急停] 停止发送速度" << std::endl;
            send_velocities = false;
        }
        if (teleop_control.is_joint_control) {
            std::cout << "[提示] 本示例未集成关节控制, 忽略关节指令" << std::endl;
        }
        
        // 检测AprilTag
        std::vector<vpHomogeneousMatrix> cMo_vec;
        detector.detect(I, config.tag_size, cam, cMo_vec);
        
        // 显示控制提示
        {
            std::stringstream ss;
            ss << "Left click to " << (send_velocities ? "stop the robot" : "servo the robot")
               << ", right click to quit.";
            vpDisplay::displayText(I, 20, 20, ss.str(), vpColor::red);
        }
        
        velocity_visual_servo = 0;
        velocity_send = 0;
        velocity_zero = 0;
        
        if (cMo_vec.size() == 1) {
            cMo = cMo_vec[0];
            
            static bool first_time = true;
            if (first_time) {
                std::vector<vpHomogeneousMatrix> v_oMo(2), v_cdMc(2);
                v_oMo[1].buildFrom(0, 0, 0, 0, 0, M_PI);
                for (size_t i = 0; i < 2; i++) {
                    v_cdMc[i] = cdMo * v_oMo[i] * cMo.inverse();
                }
                if (std::fabs(v_cdMc[0].getThetaUVector().getTheta()) <
                    std::fabs(v_cdMc[1].getThetaUVector().getTheta())) {
                    oMo = v_oMo[0];
                } else {
                    std::cout << "Desired frame modified to avoid PI rotation of the camera" << std::endl;
                    oMo = v_oMo[1];
                }
                
                for (size_t i = 0; i < point.size(); i++) {
                    vpColVector cP, p_;
                    point[i].changeFrame(cdMo * oMo, cP);
                    point[i].projection(cP, p_);
                    
                    pd[i].set_x(p_[0]);
                    pd[i].set_y(p_[1]);
                    pd[i].set_Z(cP[2]);
                }
            }
            
            std::vector<vpImagePoint> corners = detector.getPolygon(0);
            
            for (size_t i = 0; i < corners.size(); i++) {
                vpFeatureBuilder::create(p[i], cam, corners[i]);
                vpColVector cP;
                point[i].changeFrame(cMo, cP);
                p[i].set_Z(cP[2]);
            }
            
            if (config.task_sequencing) {
                if (!servo_started) {
                    if (send_velocities) {
                        servo_started = true;
                    }
                    t_init_servo = vpTime::measureTimeMs();
                }
                velocity_visual_servo = task.computeControlLaw((vpTime::measureTimeMs() - t_init_servo) / 1000.);
            } else {
                velocity_visual_servo = task.computeControlLaw();
            }
            
            // 显示特征点
            vpServoDisplay::display(task, cam, I);
            for (size_t i = 0; i < corners.size(); i++) {
                std::stringstream ss;
                ss << i;
                vpDisplay::displayText(I, corners[i] + vpImagePoint(15, 15), ss.str(), vpColor::red);
                vpImagePoint ip;
                vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), ip);
                vpDisplay::displayText(I, ip + vpImagePoint(15, 15), ss.str(), vpColor::red);
            }
            if (first_time) {
                traj_corners = new std::vector<vpImagePoint>[corners.size()];
            }
            display_point_trajectory(corners);
            
            if (config.plot) {
                plotter->plot(0, iter_plot, task.getError());
                plotter->plot(1, iter_plot, velocity_visual_servo);
                iter_plot++;
            }
            
            if (config.verbose) {
                std::cout << "velocity_visual_servo (IBVS): " << velocity_visual_servo.t();
            }
            
            double error = task.getError().sumSquare();
            {
                std::stringstream ss;
                ss << "error: " << error;
                vpDisplay::displayText(I, 20, static_cast<int>(I.getWidth()) - 150,
                                       ss.str(), vpColor::red);
            }
            
            if (config.verbose) {
                std::cout << "  error: " << error << std::endl;
            }
            
            // 收敛判定:连续 3 次 error < threshold
            if (config.convergence_threshold > 0.0 && !freeze_translation) {
                if (error < config.convergence_threshold) {
                    converged_cont += 1;
                } else {
                    converged_cont = 0;
                }
                
                if (converged_cont >= 3) {
                    freeze_translation = true;
                    std::cout << "第一阶段视觉伺服收敛:从现在起, IBVS 只控制旋转, "
                              << "键盘控制 XYZ 平移。\n";
                    vpDisplay::displayText(I, 100, 20,
                                           "Converged: Teleop controls XYZ, IBVS controls rotation",
                                           vpColor::red);
                }
            }
            
            if (first_time) {
                first_time = false;
            }
        } else {
            velocity_zero = 0;
        }
        
        // 遥操作速度命令
        vpColVector v_teleop_cmd(6, 0);
        
        if (freeze_translation && teleop_control.is_pose_control) {
            v_teleop_cmd[0] = teleop_control.pose_deltas[0];
            v_teleop_cmd[1] = teleop_control.pose_deltas[1];
            v_teleop_cmd[2] = teleop_control.pose_deltas[2];
            v_teleop_cmd[5] = teleop_control.pose_deltas[5];
        }
        
        // 合成最终速度
        if (!send_velocities) {
            velocity_send = 0;
        } else {
            if (!freeze_translation) {
                velocity_send = velocity_visual_servo;  // 第一阶段:IBVS 全 6 维
            } else {       // 第二阶段:IBVS 控制旋转, 遥操作控制平移
                // 简单的Z方向速度柔顺
                robot.getForceTorque(vpRobot::CAMERA_FRAME, force_torque);
                force_z = 0.8 * force_z + 0.2 * force_torque[2];
                if (force_z < config.force_z_threshold) {
                    v_teleop_cmd[2] -= 0.01 * (std::fabs(force_z) / std::fabs(config.force_z_threshold));
                    std::cout << "force_z < " << config.force_z_threshold << std::endl;
                }
                
                velocity_send[0] = v_teleop_cmd[0];
                velocity_send[1] = v_teleop_cmd[1];
                velocity_send[2] = v_teleop_cmd[2];
                velocity_send[3] = 0;
                velocity_send[4] = 0;
                velocity_send[5] = v_teleop_cmd[5];
            }
        }
        
        // 发送速度给机器人
        if (iter_plot >= 100) {
            robot.setVelocity(vpRobot::CAMERA_FRAME, velocity_send);
        } else {
            robot.setVelocity(vpRobot::CAMERA_FRAME, velocity_zero);
        }
        
        // 显示循环时间
        {
            std::stringstream ss;
            ss << "Loop time: " << vpTime::measureTimeMs() - t_start << " ms";
            vpDisplay::displayText(I, 40, 20, ss.str(), vpColor::red);
        }
        vpDisplay::flush(I);
        
        // 控制循环周期
        double elapsed_time_ms = vpTime::measureTimeMs() - t_start;
        double wait_time_ms = config.desired_loop_time_ms - elapsed_time_ms;
        if (wait_time_ms > 0) {
            std::this_thread::sleep_for(
                std::chrono::nanoseconds(static_cast<long long>(wait_time_ms * 1000000)));
        }
        
        // 处理鼠标事件
        vpMouseButton::vpMouseButtonType button;
        if (vpDisplay::getClick(I, button, false)) {
            switch (button) {
            case vpMouseButton::button1:
                send_velocities = !send_velocities;
                break;
            case vpMouseButton::button3:
                final_quit = true;
                velocity_send = 0;
                break;
            default:
                break;
            }
        }
    }
    
    // 停止机器人
    std::cout << "Stop the robot " << std::endl;
    robot.setRobotState(vpRobot::STATE_STOP);
    
    // 停止遥操作
    if (teleop.isRunning()) {
        teleop.stop();
    }
}

void SystemController::display_point_trajectory(const std::vector<vpImagePoint> &vip) {
    if (traj_corners == nullptr) {
        return;
    }
    
    for (size_t i = 0; i < vip.size(); i++) {
        if (traj_corners[i].size()) {
            if (vpImagePoint::distance(vip[i], traj_corners[i].back()) > 1.) {
                traj_corners[i].push_back(vip[i]);
            }
        } else {
            traj_corners[i].push_back(vip[i]);
        }
    }
    
    for (size_t i = 0; i < vip.size(); i++) {
        for (size_t j = 1; j < traj_corners[i].size(); j++) {
            vpDisplay::displayLine(I, traj_corners[i][j - 1], traj_corners[i][j], vpColor::green, 2);
        }
    }
}
