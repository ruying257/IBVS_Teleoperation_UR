#ifndef ROBOTTELEOPERATION_H
#define ROBOTTELEOPERATION_H

#include <vector>
#include <thread>
#include <termios.h>

class RobotTeleoperation {
public:
    struct ControlVector {
        std::vector<double> pose_deltas;        // 位姿变化 (m/rad)
        std::vector<double> joint_deltas;       // 关节角度变化 (rad)

        bool is_joint_control = false;          // 是否为关节控制
        bool is_pose_control  = false;          // 是否为位姿控制
        bool is_estop = false;                  // 急停标志
        bool exit_requested = false;            // 退出请求

        ControlVector();

        void zero();
    };

    RobotTeleoperation();
    ~RobotTeleoperation();

    bool start();
    void stop();

    ControlVector getControlVector() const;

    bool isRunning() const;

    void setStepSizes(
        double fine_linear = 0.005, double coarse_linear = 0.015,
        double fine_angular = 0.025, double coarse_angular = 0.12,
        double fine_joint = 0.005, double coarse_joint = 0.01
    );

    static void printInstructions();

private:
    struct KeyState {
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

        void reset_all();

        void reset();
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

#endif // ROBOTTELEOPERATION_H
