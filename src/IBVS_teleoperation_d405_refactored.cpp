/* * 重构版本: IBVS_teleoperation_d405_refactored.cpp
 * 结构优化: 引入 SystemController 类管理状态机和资源
 */

#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include <termios.h>
#include <unistd.h>
#include <fcntl.h>
#include <locale.h>
#include <csignal>
#include <cstring>

#include <visp3/core/vpConfig.h>

#if defined(VISP_HAVE_REALSENSE2) && defined(VISP_HAVE_DISPLAY) && defined(VISP_HAVE_UR_RTDE)

#include <visp3/core/vpCameraParameters.h>
#include <visp3/detection/vpDetectorAprilTag.h>
#include <visp3/gui/vpDisplayFactory.h>
#include <visp3/gui/vpPlot.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/robot/vpRobotUniversalRobots.h>
#include <visp3/sensor/vpRealSense2.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/visual_features/vpFeaturePoint.h>
#include <visp3/vs/vpServo.h>
#include <visp3/vs/vpServoDisplay.h>

#ifdef ENABLE_VISP_NAMESPACE
using namespace VISP_NAMESPACE_NAME;
#endif

// =========================================================================
// 1. 工具函数与全局配置 (Utils & Config)
// =========================================================================

// 将硬编码参数集中管理
struct AppConfig {
    std::string robot_ip = "192.168.31.100";
    double tag_size = 0.03;
    double convergence_thresh = 0.00005;
    std::string model_path = "/home/z/RemoteControl-v4/model/best.trt";

    // 相机外参 (eMc)
    vpPoseVector ePc = vpPoseVector(0.08249, 0.109173, 0.181423, -0.713955, 1.7467, -1.73549);

    // 平面估计参数
    double depth_scale = 10000.0;
};

// 辅助函数: ViSP -> OpenCV 深度图转换
cv::Mat vispDepthToCvMat(const vpImage<uint16_t>& I_depth_raw) {
    cv::Mat depth_cv(I_depth_raw.getHeight(), I_depth_raw.getWidth(), CV_16UC1, (void*)I_depth_raw.bitmap);
    return depth_cv.clone();
}

// 辅助函数: Eigen -> ViSP
vpHomogeneousMatrix eigenToVispMatrix(const Eigen::Matrix4d& T_eigen) {
    vpHomogeneousMatrix M;
    for(int i=0; i<4; i++)
        for(int j=0; j<4; j++) M[i][j] = T_eigen(i,j);
    return M;
}

// =========================================================================
// 2. 键盘遥操作类 (RobotTeleoperation) - 保持原有逻辑，建议放入 .h 文件
// =========================================================================
class RobotTeleoperation {
public:
    struct ControlVector {
        std::vector<double> pose_deltas;        // 位姿变化 (m/rad)
        std::vector<double> joint_deltas;       // 关节角度变化 (rad)

        bool is_joint_control = false;          // 是否为关节控制
        bool is_pose_control  = false;          // 是否为位姿控制
        bool is_estop = false;                  // 急停标志
        bool exit_requested = false;            // 退出请求

        ControlVector() : pose_deltas(6, 0.0), joint_deltas(6, 0.0) {}

        void zero() {
            std::fill(pose_deltas.begin(), pose_deltas.end(), 0.0);
            std::fill(joint_deltas.begin(), joint_deltas.end(), 0.0);
        }
    };
    RobotTeleoperation();
    ~RobotTeleoperation();
    bool start();
    void stop();
    ControlVector getControlVector() const;
    void setStepSizes(
        double fine_linear = 0.005, double coarse_linear = 0.015,
        double fine_angular = 0.025, double coarse_angular = 0.12,
        double fine_joint = 0.005, double coarse_joint = 0.01
    );
    static void printInstructions();

private:
    struct KeyState {
        /* 用于存储键盘状态 */
        // 平移控制
        bool w = false; bool x = false;
        bool a = false; bool d = false;
        bool r = false; bool f = false;
        bool s = false;

        // 旋转控制
        bool i = false; bool k = false;
        bool j = false; bool l = false;
        bool u = false; bool o = false;

        // 关节控制
        bool num1 = false; bool num2 = false; bool num3 = false;
        bool num4 = false; bool num5 = false; bool num6 = false;
        bool num1_shift = false; bool num2_shift = false; bool num3_shift = false;
        bool num4_shift = false; bool num5_shift = false; bool num6_shift = false;

        // 功能键
        bool fine = false;      // 精细模式
        bool exit_flag = false; // 退出
        bool estop = false;     // 急停

        void reset_all() {
            w = x = s = a = d = r = f = false;
            i = k = j = l = u = o = false;
            num1 = num2 = num3 = num4 = num5 = num6 = false;
            num1_shift = num2_shift = num3_shift = num4_shift = num5_shift = num6_shift = false;
            fine = exit_flag = estop = false;
        }

        void reset() {
            w = x = s = a = d = r = f = false;
            i = k = j = l = u = o = false;
            num1 = num2 = num3 = num4 = num5 = num6 = false;
            num1_shift = num2_shift = num3_shift = num4_shift = num5_shift = num6_shift = false;
        }
    };

    void keyboardThread();
    bool initTerminal();
    void restoreTerminal();

    double   fine_linear_step  = 0.001;
    double coarse_linear_step  = 0.005;
    double   fine_angular_step = 0.005;
    double coarse_angular_step = 0.01;
    double   fine_joint_step   = 0.005;
    double coarse_joint_step   = 0.01;

    bool running = false;
    KeyState key_state;
    std::thread keyboard_thread;
    struct termios old_tio, new_tio;

    RobotTeleoperation(const RobotTeleoperation&) = delete;
    RobotTeleoperation& operator=(const RobotTeleoperation&) = delete;
};

// =========================================================================
// 3. 核心系统控制器 (SystemController)
// =========================================================================

class SystemController {
public:
    enum State {
        STATE_IBVS,           // 视觉伺服阶段
        STATE_WAIT_SELECT,    // 等待选择 (YOLO/Manual)
        STATE_APPROACH,       // 执行平面逼近 (动作执行)
        STATE_TELEOP          // 纯遥操模式
    };

    SystemController(const AppConfig& conf) : config(conf) {}
    ~SystemController() { cleanup(); }

    // 初始化硬件和算法
    bool initialize() {
        try {
            // 1. 连接机器人
            robot.connect(config.robot_ip);
            robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);
            move_to_safe_pose();

            // 2. 初始化相机
            rs2::config rs_cfg;
            rs_cfg.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_RGBA8, 30);
            rs_cfg.enable_stream(RS2_STREAM_DEPTH, 1280, 720, RS2_FORMAT_Z16, 30);
            rs_cfg.disable_stream(RS2_STREAM_INFRARED);
            rs.open(rs_cfg);

            // 获取内参
            cam = rs.getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithDistortion);

            // 设置外参
            vpHomogeneousMatrix eMc(config.ePc);
            robot.set_eMc(eMc);

            // 3. 初始化算法模块
            yolo_detector = std::make_unique<TensorRT_detection>(config.model_path);
            setup_plane_estimator();
            setup_ibvs_task();
            setup_display();

            // 4. 启动键盘控制
            teleop.start();
            RobotTeleoperation::printInstructions();

            return true;
        } catch (const std::exception& e) {
            std::cerr << "Init Error: " << e.what() << std::endl;
            return false;
        }
    }

    // 主循环
    void run() {
        bool quit = false;
        while (!quit) {
            auto t_start = vpTime::measureTimeMs();

            // 1. 获取数据
            acquire_images();
            teleop_cmd = teleop.getControlVector();

            // 2. 状态处理与全局控制
            if (teleop_cmd.exit_requested) quit = true;
            handle_ui_interaction(quit); // 处理鼠标点击等

            // 3. 核心状态机逻辑
            vpColVector v_cmd(6, 0);

            switch (current_state) {
                case STATE_IBVS:
                    v_cmd = process_state_ibvs();
                    break;
                case STATE_WAIT_SELECT:
                    process_state_selection(); // 此状态通常不发送连续速度，而是等待触发动作
                    v_cmd = 0;
                    break;
                case STATE_APPROACH:
                    // 过渡状态，通常瞬间完成
                    current_state = STATE_TELEOP;
                    break;
                case STATE_TELEOP:
                    v_cmd = process_state_teleop();
                    break;
            }

            // 4. 发送速度 (安全检查)
            send_velocity(v_cmd);

            // 5. 绘图与刷新
            update_display();

            // 6. 循环控频 (20ms)
            enforce_loop_rate(t_start, 20.0);
        }
    }

private:
    // --- 成员变量 ---
    AppConfig config;
    vpRobotUniversalRobots robot;
    vpRealSense2 rs;
    vpCameraParameters cam;
    RobotTeleoperation teleop;
    RobotTeleoperation::ControlVector teleop_cmd;

    // 视觉与显示
    vpImage<vpRGBa> I_color;
    vpImage<uint16_t> I_depth;
    vpImage<unsigned char> I_gray;
    std::unique_ptr<vpDisplay> display;

    // 算法对象
    std::unique_ptr<TensorRT_detection> yolo_detector;
    std::unique_ptr<PlaneEstimator> plane_estimator; // 假设有默认构造或指针管理
    vpDetectorAprilTag detector_tag;
    vpServo task_ibvs;

    // 状态管理
    State current_state = STATE_IBVS;
    bool send_velocities = false;
    int ibvs_converged_count = 0;

    // IBVS 特征点
    std::vector<vpFeaturePoint> s_current, s_desired;
    std::vector<vpPoint> points_3d;

    // --- 私有辅助方法 ---

    void move_to_safe_pose() {
        vpColVector q(6);
        q << vpMath::rad(152.65), vpMath::rad(-110.89), vpMath::rad(119.46),
             vpMath::rad(-103.64), vpMath::rad(90.38), vpMath::rad(-107.6);
        robot.setRobotState(vpRobot::STATE_POSITION_CONTROL);
        robot.setPosition(vpRobot::JOINT_STATE, q);
        robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);
    }

    void setup_plane_estimator() {
        PlaneEstimator::CameraIntrinsics intr;
        intr.fx = cam.get_px(); intr.fy = cam.get_py();
        intr.cx = cam.get_u0(); intr.cy = cam.get_v0();
        PlaneEstimator::Params p_params;
        // ... (填充参数) ...
        plane_estimator = std::make_unique<PlaneEstimator>(intr, config.depth_scale, p_params);
    }

    void setup_ibvs_task() {
        // 初始化 AprilTag 检测器和 IBVS 任务
        detector_tag.setAprilTagQuadDecimate(2);
        task_ibvs.setServo(vpServo::EYEINHAND_CAMERA);
        task_ibvs.setInteractionMatrixType(vpServo::CURRENT);
        task_ibvs.setLambda(0.5);

        // 初始化特征点 (4个点)
        s_current.resize(4); s_desired.resize(4); points_3d.resize(4);
        // 设置 points_3d 坐标和 s_desired ... (保持原逻辑)
    }

    void setup_display() {
        I_color.resize(720, 1280);
        I_depth.resize(720, 1280);
        I_gray.resize(720, 1280);
        display = vpDisplayFactory::createDisplay(I_color, 10, 10, "Robot View");
    }

    void acquire_images() {
        rs.acquire((unsigned char *)I_color.bitmap, (unsigned char *)I_depth.bitmap, nullptr, nullptr, nullptr, nullptr);
        vpDisplay::display(I_color);
        vpImageConvert::convert(I_color, I_gray);
    }

    // --- 状态处理函数 ---

    vpColVector process_state_ibvs() {
        std::vector<vpHomogeneousMatrix> cMo_vec;
        detector_tag.detect(I_gray, config.tag_size, cam, cMo_vec);

        vpColVector v_ibvs(6, 0);

        if (cMo_vec.size() > 0) {
            // 更新特征点 s_current ...
            // 计算控制律
            v_ibvs = task_ibvs.computeControlLaw();

            // 绘制特征点 ...
            vpServoDisplay::display(task_ibvs, cam, I_color);

            // 收敛判断
            if (task_ibvs.getError().sumSquare() < config.convergence_thresh) {
                ibvs_converged_count++;
                if (ibvs_converged_count > 3) {
                    std::cout << "[Info] IBVS Converged. Switching to Selection." << std::endl;
                    current_state = STATE_WAIT_SELECT;
                    send_velocities = false; // 暂停运动
                }
            } else {
                ibvs_converged_count = 0;
            }
        }
        return v_ibvs;
    }

    void process_state_selection() {
        vpDisplay::displayText(I_color, 40, 20, "MODE: SELECT. Press 'Y'(Auto) or 'M'(Click)", vpColor::yellow);

        // 简单的输入映射 (借用 Joint 按键或鼠标)
        bool trigger_yolo = teleop_cmd.joint_deltas[0] != 0;

        if (trigger_yolo) {
             // 执行 YOLO 检测 -> Plane Estimator -> robot.setPosition
             // ... (这里填入原代码中 STATE_WAIT_SELECT 的逻辑) ...

             // 成功后:
             current_state = STATE_TELEOP;
        }
    }

    vpColVector process_state_teleop() {
        vpDisplay::displayText(I_color, 40, 20, "MODE: TELEOP", vpColor::green);

        vpColVector v_cmd(6, 0);
        // 读取键盘平移指令
        if (teleop_cmd.is_pose_control) {
             v_cmd[0] = teleop_cmd.pose_deltas[0];
             v_cmd[1] = teleop_cmd.pose_deltas[1];
             v_cmd[2] = teleop_cmd.pose_deltas[2];
             v_cmd[5] = teleop_cmd.pose_deltas[5]; // Rz
        }

        // 力控安全保护逻辑
        vpColVector ft(6);
        robot.getForceTorque(vpRobot::CAMERA_FRAME, ft);
        if (ft[2] < -15.0) {
             v_cmd[2] -= 0.01; // 回退
             vpDisplay::displayText(I_color, 60, 20, "FORCE LIMIT!", vpColor::red);
        }

        return v_cmd;
    }

    void send_velocity(const vpColVector& v) {
        if (send_velocities && !teleop_cmd.is_estop) {
            robot.setVelocity(vpRobot::CAMERA_FRAME, v);
        } else {
            robot.setVelocity(vpRobot::CAMERA_FRAME, vpColVector(6, 0));
        }
    }

    void handle_ui_interaction(bool& quit_flag) {
        vpMouseButton::vpMouseButtonType button;
        if (vpDisplay::getClick(I_color, button, false)) {
            if (button == vpMouseButton::button1) send_velocities = !send_velocities;
            if (button == vpMouseButton::button3) quit_flag = true;
        }
        // 显示当前状态文字
        std::string status = send_velocities ? "Robot: ACTIVE" : "Robot: PAUSED";
        vpDisplay::displayText(I_color, 20, 20, status, send_velocities ? vpColor::green : vpColor::red);
    }

    void update_display() {
        vpDisplay::flush(I_color);
    }

    void cleanup() {
        robot.setRobotState(vpRobot::STATE_STOP);
        teleop.stop();
        std::cout << "System Cleaned up." << std::endl;
    }

    void enforce_loop_rate(double t_start, double period_ms) {
        double elapsed = vpTime::measureTimeMs() - t_start;
        if (elapsed < period_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds((long)(period_ms - elapsed)));
        }
    }
};

// =========================================================================
// 4. 程序入口 (Main Entry)
// =========================================================================

int main(int argc, char** argv) {
    // 1. 解析参数 (这里简化处理，可以使用 argparse 库)
    AppConfig config;
    // if(argc > 1) config.robot_ip = argv[1]; ...

    // 2. 实例化控制器
    SystemController app(config);

    // 3. 初始化并运行
    if (app.initialize()) {
        app.run();
    } else {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
