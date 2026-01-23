/*
 * 集成版本:
 * - ViSP IBVS 控制 UR12e (眼在手上)
 * - 键盘遥操作 RobotTeleoperation
 *
 * 行为逻辑:
 *   收敛前:IBVS 控制 6 维速度；
 *   第一次收敛后:
 *     - IBVS 只控制 wx, wy, wz(后三维角速度)
 *     - 遥操作只控制 vx, vy, vz(前三维线速度)
 *
 * 抖动处理:
 *   - 遥操作线速度采用一阶低通滤波, 避免 0 / 非 0 突跳；
 *   - 控制周期从 50 ms 改为 20 ms；
 *   - 遥操作增益 teleop_velocity_gain = 2.0。
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



// ====================================================================================================================
// ==========================================                                ==========================================
// ========================================== 0. RobotTeleoperation 定义部分 ==========================================
// ==========================================                                ==========================================
// ====================================================================================================================
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

    bool isRunning() const { return running; }

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


// ====================================================================================================================
// =========================================                                ===========================================
// ========================================= 1. RobotTeleoperation 实现部分 ===========================================
// =========================================                                ===========================================
// ====================================================================================================================

// 全局信号标志
static volatile bool g_signal_received = false;

/*************************************************
 *
 * @brief   信号处理函数
 *
 * @details 该函数在接收到指定信号时被调用,
 *          设置全局标志以通知程序退出.
 *
 * @param   signum 接收到的信号编号
 *
 *************************************************/
void signalHandler(int signum) {
  g_signal_received = true;
  std::cout << "\n收到信号 " << signum << ", 正在退出..." << std::endl;
}

RobotTeleoperation::RobotTeleoperation() {
  setlocale(LC_ALL, "zh_CN.UTF-8");
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);
  key_state.reset();
}

RobotTeleoperation::~RobotTeleoperation() {
  stop();
}

/*************************************************
 *
 *
 *
 *************************************************/
bool RobotTeleoperation::start() {
  if (running) {
    std::cout << "键盘监听已在运行中" << std::endl;
    return true;
  }

  if (!initTerminal()) {
    std::cerr << "终端初始化失败" << std::endl;
    return false;
  }

  key_state.reset();
  running = true;
  keyboard_thread = std::thread(&RobotTeleoperation::keyboardThread, this);

  std::cout << "键盘监听已启动" << std::endl;
  return true;
}

/*************************************************
 *
 *
 *
 *************************************************/
void RobotTeleoperation::stop() {
  if (!running) {
    return;
  }

  running = false;

  if (keyboard_thread.joinable()) {
    keyboard_thread.join();
  }

  restoreTerminal();
  std::cout << "键盘监听已停止" << std::endl;
}

/*************************************************
 *
 * @brief   获取当前控制向量
 *
 * @details 根据当前按键状态生成控制向量
 *
 * @return  包含位姿和关节变化的控制向量
 *
 *************************************************/
RobotTeleoperation::ControlVector RobotTeleoperation::getControlVector() const {
  ControlVector control;

  if (key_state.exit_flag) {
    control.exit_requested = true;
    return control;
  }

  control.is_estop = key_state.estop;
  if (key_state.estop) {
    control.is_estop = true;
    return control;
  }

  bool fine_mode = key_state.fine;
  double linear_step  = fine_mode ? fine_linear_step  : coarse_linear_step;
  double angular_step = fine_mode ? fine_angular_step : coarse_angular_step;
  double joint_step   = fine_mode ? fine_joint_step   : coarse_joint_step;

  // 关节控制
  control.joint_deltas[0] = key_state.num1 ? joint_step : (key_state.num1_shift ? -joint_step : 0);
  control.joint_deltas[1] = key_state.num2 ? joint_step : (key_state.num2_shift ? -joint_step : 0);
  control.joint_deltas[2] = key_state.num3 ? joint_step : (key_state.num3_shift ? -joint_step : 0);
  control.joint_deltas[3] = key_state.num4 ? joint_step : (key_state.num4_shift ? -joint_step : 0);
  control.joint_deltas[4] = key_state.num5 ? joint_step : (key_state.num5_shift ? -joint_step : 0);
  control.joint_deltas[5] = key_state.num6 ? joint_step : (key_state.num6_shift ? -joint_step : 0);
  if (key_state.s) { control.joint_deltas = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; } // 's' 键清除关节命令

  for (double delta : control.joint_deltas) {
    if (delta != 0.0) {
      control.is_joint_control = true;
      return control;
    }
  }

  // 位姿控制
  control.pose_deltas[0] = key_state.d ? linear_step : (key_state.a ? -linear_step : 0);
  control.pose_deltas[1] = key_state.x ? linear_step : (key_state.w ? -linear_step : 0);
  control.pose_deltas[2] = key_state.r ? linear_step : (key_state.f ? -linear_step : 0);
  control.pose_deltas[3] = key_state.i ? angular_step : (key_state.k ? -angular_step : 0);
  control.pose_deltas[4] = key_state.j ? angular_step : (key_state.l ? -angular_step : 0);
  control.pose_deltas[5] = key_state.u ? angular_step : (key_state.o ? -angular_step : 0);
  if (key_state.s) { control.pose_deltas = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; } // 's' 键清除位姿命令

  for (double delta : control.pose_deltas) {
    if (delta != 0.0) {
      control.is_pose_control = true;
      return control;
    }
  }

  return control;
}

/*************************************************
 *
 * @brief   设置遥操作单步步长
 *
 * @details 设置精细和快速模式下的线性、角度和关节步长
 *
 *************************************************/
void RobotTeleoperation::setStepSizes(
  double fine_linear, double coarse_linear,
  double fine_angular, double coarse_angular,
  double fine_joint, double coarse_joint
) {
  fine_linear_step    = fine_linear;
  coarse_linear_step  = coarse_linear;
  fine_angular_step   = fine_angular;
  coarse_angular_step = coarse_angular;
  fine_joint_step     = fine_joint;
  coarse_joint_step   = coarse_joint;
}

/*************************************************
 *
 *
 *
 *************************************************/
void RobotTeleoperation::printInstructions() {
  std::cout << "\n===== 机械臂键盘控制说明 =====" << std::endl;
  std::cout << "平移控制:W(+X) X(-X) S(Stop) A(-Y) D(+Y) R(+Z) F(-Z)" << std::endl;
  std::cout << "旋转控制:I(+Rx) K(-Rx) J(+Ry) L(-Ry) U(+Rz) O(-Rz)" << std::endl;
  std::cout << "关节控制:1-6(正转) !@#$%^(反转, Shift+数字)" << std::endl;
  std::cout << "模式控制:按住Z键=精细模式  空格键=急停  Q键=退出" << std::endl;
  std::cout << "=====================================" << std::endl;
  std::cout << "提示:松开按键后机械臂会立即停止运动" << std::endl;
}

/*************************************************
 *
 * @brief   键盘监听线程
 *
 * @details 该函数在独立线程中运行, 持续监听键盘输入,
 *          并根据按键状态更新控制向量.
 *
 * @note    该函数会持续运行直到收到退出信号.
 *************************************************/
void RobotTeleoperation::keyboardThread() {
  std::cout << "键盘监听线程开始运行" << std::endl;

  while (running) {
    if (g_signal_received) {
      key_state.exit_flag = true;
      break;
    }

    char ch;
    ssize_t n = read(STDIN_FILENO, &ch, 1);

    if (n > 0) {
      // 保留状态切换的键位, 其他键位重置
      key_state.reset();

      switch (ch) {
      case 'q':
      case 'Q':
        key_state.exit_flag = true;
        break;

      case ' ':
        key_state.estop = !key_state.estop;
        if (!key_state.estop) {std::cout << "[急停已解除] 请用鼠标左键点击画面恢复发送速度" << std::endl;}
        break;

      case 'z':
      case 'Z':
        key_state.fine = !key_state.fine;
        std::cout << (key_state.fine ? "进入精细模式" : "退出精细模式") << std::endl;
        break;

      // 平移
      case 'w':
      case 'W':
        key_state.w = true;
        break;
      case 'x':
      case 'X':
        key_state.x = true;
        break;
      case 's':
      case 'S':
        key_state.s = true;
        break;
      case 'a':
      case 'A':
        key_state.a = true;
        break;
      case 'd':
      case 'D':
        key_state.d = true;
        break;
      case 'r':
      case 'R':
        key_state.r = true;
        break;
      case 'f':
      case 'F':
        key_state.f = true;
        break;

      // 旋转
      case 'i':
      case 'I':
        key_state.i = true;
        break;
      case 'k':
      case 'K':
        key_state.k = true;
        break;
      case 'j':
      case 'J':
        key_state.j = true;
        break;
      case 'l':
      case 'L':
        key_state.l = true;
        break;
      case 'u':
      case 'U':
        key_state.u = true;
        break;
      case 'o':
      case 'O':
        key_state.o = true;
        break;

      // 关节正转
      case '1':
        key_state.num1 = true;
        break;
      case '2':
        key_state.num2 = true;
        break;
      case '3':
        key_state.num3 = true;
        break;
      case '4':
        key_state.num4 = true;
        break;
      case '5':
        key_state.num5 = true;
        break;
      case '6':
        key_state.num6 = true;
        break;

      // 关节反转(Shift+数字)
      case '!':
        key_state.num1_shift = true;
        break;
      case '@':
        key_state.num2_shift = true;
        break;
      case '#':
        key_state.num3_shift = true;
        break;
      case '$':
        key_state.num4_shift = true;
        break;
      case '%':
        key_state.num5_shift = true;
        break;
      case '^':
        key_state.num6_shift = true;
        break;

      default:
        break;
      }
    }

    usleep(5000); // 5ms
  }

  std::cout << "键盘监听线程结束" << std::endl;
}

bool RobotTeleoperation::initTerminal() {
  // 获取当前终端设置
  if (tcgetattr(STDIN_FILENO, &old_tio) == -1) {
    std::cerr << "获取终端设置失败" << std::endl;
    return false;
  }

  // 复制一份用于修改
  new_tio = old_tio;

  // 禁用行缓冲, 但保留回显
  new_tio.c_lflag &= ~(ICANON);  // 仅关闭 ICANON
  new_tio.c_lflag |= ECHO;       // 显式开启回显

  // 非阻塞读取配置
  new_tio.c_cc[VMIN]  = 0;
  new_tio.c_cc[VTIME] = 0;

  if (tcsetattr(STDIN_FILENO, TCSANOW, &new_tio) == -1) {
    std::cerr << "设置终端失败" << std::endl;
    return false;
  }

  int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  if (flags == -1) {
    std::cerr << "获取文件标志失败" << std::endl;
    return false;
  }

  if (fcntl(STDIN_FILENO, F_SETFL, flags | O_NONBLOCK) == -1) {
    std::cerr << "设置非阻塞失败" << std::endl;
    return false;
  }

  return true;
}

void RobotTeleoperation::restoreTerminal() {
  // 恢复终端设置
  tcsetattr(STDIN_FILENO, TCSANOW, &old_tio);

  // 恢复阻塞模式
  int flags = fcntl(STDIN_FILENO, F_GETFL, 0);
  if (flags != -1) {
    fcntl(STDIN_FILENO, F_SETFL, flags & ~O_NONBLOCK);
  }
}


// =============================================================================================================
// ==========================================                         ==========================================
// ========================================== 2. IBVS + Teleop 主程序 ==========================================
// ==========================================                         ==========================================
// =============================================================================================================

void display_point_trajectory(
  const vpImage<unsigned char> &I,
  const std::vector<vpImagePoint> &vip,
  std::vector<vpImagePoint> *traj_vip
) {
  for (size_t i = 0; i < vip.size(); i++) {
    if (traj_vip[i].size()) {
      if (vpImagePoint::distance(vip[i], traj_vip[i].back()) > 1.) {
        traj_vip[i].push_back(vip[i]);
      }
    }
    else {
      traj_vip[i].push_back(vip[i]);
    }
  }
  for (size_t i = 0; i < vip.size(); i++) {
    for (size_t j = 1; j < traj_vip[i].size(); j++) {
      vpDisplay::displayLine(I, traj_vip[i][j - 1], traj_vip[i][j], vpColor::green, 2);
    }
  }
}

int main(int argc, char **argv) {
  /* 设置默认参数值 */
  double opt_tagSize = 0.03;                    // AprilTag标签尺寸, 单位米
  std::string opt_robot_ip = "192.168.31.100";  // UR12e机器人IP
  std::string opt_eMc_filename = "";            // 外参文件名, 默认为空
  bool display_tag = true;                      // 是否在图像上显示检测到的标签
  int opt_quad_decimate = 2;                    // AprilTag检测时的图像降采样因子(2表示分辨率减半)
  bool opt_verbose = false;                     // 是否输出详细调试信息
  bool opt_plot = false;                        // 是否实时绘制曲线图
  bool opt_adaptive_gain = false;               // 是否使用自适应增益
  bool opt_task_sequencing = false;             // 是否使用任务序列化(时间相关的控制律)
  double convergence_threshold = 0.00005;       // 收敛阈值, 误差小于此值时认为伺服完成
  int converged_cont = 0;                       // 连续收敛计数器

  /* 解析命令行参数 */
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--tag-size" && i + 1 < argc) {
      opt_tagSize = std::stod(argv[++i]);
    }
    else if (std::string(argv[i]) == "--tag-quad-decimate" && i + 1 < argc) {
      opt_quad_decimate = std::stoi(argv[++i]);
    }
    else if (std::string(argv[i]) == "--ip" && i + 1 < argc) {
      opt_robot_ip = std::string(argv[++i]);
    }
    else if (std::string(argv[i]) == "--eMc" && i + 1 < argc) {
      opt_eMc_filename = std::string(argv[++i]);
    }
    else if (std::string(argv[i]) == "--verbose") {
      opt_verbose = true;
    }
    else if (std::string(argv[i]) == "--plot") {
      opt_plot = true;
    }
    else if (std::string(argv[i]) == "--adpative-gain") {
      opt_adaptive_gain = true;
    }
    else if (std::string(argv[i]) == "--task-sequencing") {
      opt_task_sequencing = true;
    }
    else if (std::string(argv[i]) == "--no-convergence-threshold") {
      convergence_threshold = 0.;
    }
    else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      std::cout
        << argv[0] << " [--ip <default " << opt_robot_ip << ">] [--tag-size <marker size in meter; default "
        << opt_tagSize << ">] [--eMc <eMc extrinsic file>] "
        << "[--tag-quad-decimate <decimation; default " << opt_quad_decimate
        << ">] [--adpative-gain] [--plot] [--task-sequencing] [--no-convergence-threshold] [--verbose] [--help] [-h]"
        << "\n";
      return EXIT_SUCCESS;
    }
  }

  vpRobotUniversalRobots robot;
  RobotTeleoperation teleop;

#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
  std::shared_ptr<vpDisplay> display;
#else
  vpDisplay *display = nullptr;
#endif

  try {
    robot.connect(opt_robot_ip);

    std::cout << "WARNING: This example will move the robot! "
              << "Please make sure to have the user stop button at hand!" << std::endl
              << "Press Enter to continue..." << std::endl;
    std::cin.ignore();

    /* 晚安楼D405的初始关节位姿 */
    vpColVector q(6, 0);  // 初始化安全位姿(关节位姿)
    q[0] =  vpMath::rad(152.65);
    q[1] = -vpMath::rad(110.89);
    q[2] =  vpMath::rad(119.46);
    q[3] = -vpMath::rad(103.64);
    q[4] =  vpMath::rad(90.38);
    q[5] = -vpMath::rad(107.6);
    std::cout << "Move to joint position: " << q.t() << std::endl;
    robot.setRobotState(vpRobot::STATE_POSITION_CONTROL);
    robot.setPosition(vpRobot::JOINT_STATE, q);

    /* realsense D405相机实例及配置 */
    vpRealSense2 rs;
    rs2::config config;
    unsigned int width = 1280, height = 720;
    config.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_RGBA8, 30);
    config.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16,   30);
    config.disable_stream(RS2_STREAM_INFRARED);  //D405没有红外流
    rs.open(config);

    /* realsense D405 相机外参设置 */
    vpPoseVector ePc(0, 0, 0, 0, 0, 0);  // 相机外参初始化为全0
    // ePc[0] =  0.141335;
    // ePc[1] =  0.174283;
    // ePc[2] =  0.0915465;
    // ePc[3] = -0.556397;
    // ePc[4] =  1.79421;
    // ePc[5] = -1.77789;

    if (!opt_eMc_filename.empty()) {
      ePc.loadYAML(opt_eMc_filename, ePc);
    }
    else {
      std::cout << "Warning, opt_eMc_filename is empty! Use hard coded values.\n";
    }
    vpHomogeneousMatrix eMc(ePc);
    std::cout << "eMc:\n" << eMc << "\n";

    /* realsense D405 相机内参设置 */
    vpCameraParameters cam =
      rs.getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithDistortion);
    std::cout << "cam:\n" << cam << "\n";

    vpImage<unsigned char> I(height, width);

#if (VISP_CXX_STANDARD >= VISP_CXX_STANDARD_11)
    display = vpDisplayFactory::createDisplay(I, 10, 10, "Color image");
#else
    display = vpDisplayFactory::allocateDisplay(I, 10, 10, "Color image");
#endif

    vpDetectorAprilTag::vpAprilTagFamily tagFamily = vpDetectorAprilTag::TAG_36h11;
    vpDetectorAprilTag::vpPoseEstimationMethod poseEstimationMethod =
      vpDetectorAprilTag::HOMOGRAPHY_VIRTUAL_VS;
    vpDetectorAprilTag detector(tagFamily);
    detector.setAprilTagPoseEstimationMethod(poseEstimationMethod);
    detector.setDisplayTag(display_tag);
    detector.setAprilTagQuadDecimate(opt_quad_decimate);

    vpHomogeneousMatrix cdMc, cMo, oMo;
    vpHomogeneousMatrix cdMo(vpTranslationVector(0, 0, opt_tagSize * 10),  // 原本是opt_tagSize * 3
                             vpRotationMatrix({ 1, 0, 0, 0, -1, 0, 0, 0, -1 }));

    std::vector<vpFeaturePoint> p(4), pd(4);
    std::vector<vpPoint> point(4);
    point[0].setWorldCoordinates(-opt_tagSize / 2., -opt_tagSize / 2., 0);
    point[1].setWorldCoordinates( opt_tagSize / 2., -opt_tagSize / 2., 0);
    point[2].setWorldCoordinates( opt_tagSize / 2.,  opt_tagSize / 2., 0);
    point[3].setWorldCoordinates(-opt_tagSize / 2.,  opt_tagSize / 2., 0);

    vpServo task;
    for (size_t i = 0; i < p.size(); i++) {
      task.addFeature(p[i], pd[i]);
    }
    task.setServo(vpServo::EYEINHAND_CAMERA);
    task.setInteractionMatrixType(vpServo::CURRENT);

    if (opt_adaptive_gain) {
      vpAdaptiveGain lambda(1.5, 0.4, 30);
      task.setLambda(lambda);
    }
    else {
      task.setLambda(0.5);
    }

    vpPlot *plotter = nullptr;
    int iter_plot = 0;

    if (opt_plot) {
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

    bool final_quit = false;
    bool send_velocities = false;     // 鼠标左键控制是否发送速度
    bool servo_started = false;
    std::vector<vpImagePoint> *traj_corners = nullptr;

    static double t_init_servo = vpTime::measureTimeMs();

    bool freeze_translation = false;  // 第一阶段收敛后:冻结 IBVS, 改为遥操作控制

    robot.set_eMc(eMc);
    robot.setRobotState(vpRobot::STATE_VELOCITY_CONTROL);

    /* 启动键盘遥操作 */
    RobotTeleoperation::printInstructions();
    teleop.setStepSizes(
      0.005, 0.015,  // 平移精细/快速
      0.025, 0.120,  // 旋转精细/快速(本例仅pose_deltas[5]使用)
      0.005, 0.010   // 关节精细/快速(本例未用)
    );
    if (!teleop.start()) {
      std::cerr << "无法启动键盘监听" << std::endl;
      return EXIT_FAILURE;
    }
    std::cout << "\n键盘遥操作已启动:收敛前为 IBVS 全 6 维控制；"
              << "收敛后:键盘控制 XYZ, 取消除rz以外的旋转。\n"
              << "按 Q 退出, 空格急停, 鼠标左键开/关速度发送, 右键退出程序。\n";

    /* 力传感器数据 */
    vpColVector force_torque(6, 0);
    robot.zeroFTSensor();
    robot.getForceTorque(vpRobot::CAMERA_FRAME, force_torque);
    double force_z = force_torque[2];  // 沿相机Z轴方向的力

    const double desired_loop_time_ms = 20.0;
    double elapsed_time_ms;
    double wait_time_ms;

    vpColVector velocity_visual_servo(6, 0);  // IBVS 输出速度
    vpColVector velocity_send(6, 0);   // 最终发送速度
    vpColVector velocity_zero(6, 0);   // 零速度
    // ================================================================================================
    // ==========================================            ==========================================
    // ========================================== 主循环开始 ==========================================
    // ==========================================            ==========================================
    // ================================================================================================
    while (!final_quit) {
      double t_start = vpTime::measureTimeMs();

      /* 从相机获取并显示图像 */
      rs.acquire(I);
      vpDisplay::display(I);

      /* 读取键盘控制 */
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

      std::vector<vpHomogeneousMatrix> cMo_vec;
      detector.detect(I, opt_tagSize, cam, cMo_vec);

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
          }
          else {
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

        if (opt_task_sequencing) {
          if (!servo_started) {
            if (send_velocities) {
              servo_started = true;
            }
            t_init_servo = vpTime::measureTimeMs();
          }
          velocity_visual_servo = task.computeControlLaw((vpTime::measureTimeMs() - t_init_servo) / 1000.);
        }
        else {
          velocity_visual_servo = task.computeControlLaw();
        }

        /* 显示特征点 */
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
        display_point_trajectory(I, corners, traj_corners);

        if (opt_plot) {
          plotter->plot(0, iter_plot, task.getError());
          plotter->plot(1, iter_plot, velocity_visual_servo);
          iter_plot++;
        }

        if (opt_verbose) {
          std::cout << "velocity_visual_servo (IBVS): " << velocity_visual_servo.t();
        }

        double error = task.getError().sumSquare();
        {
          std::stringstream ss;
          ss << "error: " << error;
          vpDisplay::displayText(I, 20, static_cast<int>(I.getWidth()) - 150,
                                 ss.str(), vpColor::red);
        }

        if (opt_verbose) {
          std::cout << "  error: " << error << std::endl;
        }

        /* 收敛判定:连续 3 次 error < threshold */
        if (convergence_threshold > 0.0 && !freeze_translation) {
          if (error < convergence_threshold) {
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
      }
      else {
        velocity_zero = 0;
      }

      /* 遥操作速度命令(只在 freeze_translation 之后启用) */
      vpColVector v_teleop_cmd(6, 0);

      if (freeze_translation && teleop_control.is_pose_control) {
        v_teleop_cmd[0] = teleop_control.pose_deltas[0];
        v_teleop_cmd[1] = teleop_control.pose_deltas[1];
        v_teleop_cmd[2] = teleop_control.pose_deltas[2];
        v_teleop_cmd[5] = teleop_control.pose_deltas[5];
      }

      /* 合成最终速度 */
      if (!send_velocities) {
        velocity_send = 0;
      } else {
        if (!freeze_translation) {
          velocity_send = velocity_visual_servo;  // 第一阶段:IBVS 全 6 维
        } else {       // 第二阶段:IBVS 控制旋转, 遥操作控制平移
          /* 简单的Z方向速度柔顺 */
          robot.getForceTorque(vpRobot::CAMERA_FRAME, force_torque);
          force_z = 0.8 * force_z + 0.2 * force_torque[2];
          if (force_z < -15.0){
            v_teleop_cmd[2] -= 0.01 * (std::fabs(force_z) / 15.0);
            std::cout << "force_z < -15.0";
          }

          velocity_send[0] = v_teleop_cmd[0];
          velocity_send[1] = v_teleop_cmd[1];
          velocity_send[2] = v_teleop_cmd[2];
          velocity_send[3] = 0;
          velocity_send[4] = 0;
          velocity_send[5] = v_teleop_cmd[5];
        }
      }

      /* 发送给机器人(前 100 次迭代保持静止, 避免启动瞬态影响) */
      if (iter_plot >= 100) {
        robot.setVelocity(vpRobot::CAMERA_FRAME, velocity_send);
      } else {
        robot.setVelocity(vpRobot::CAMERA_FRAME, velocity_zero);
      }

      {
        std::stringstream ss;
        ss << "Loop time: " << vpTime::measureTimeMs() - t_start << " ms";
        vpDisplay::displayText(I, 40, 20, ss.str(), vpColor::red);
      }
      vpDisplay::flush(I);

      /* 控制循环周期:20 ms */
      elapsed_time_ms = vpTime::measureTimeMs() - t_start;
      wait_time_ms    = desired_loop_time_ms - elapsed_time_ms;
      if (wait_time_ms > 0) {
        std::this_thread::sleep_for(
          std::chrono::nanoseconds(static_cast<long long>(wait_time_ms * 1000000)));
      }

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
    // ================================================================================================
    // ==========================================            ==========================================
    // ========================================== 主循环结束 ==========================================
    // ==========================================            ==========================================
    // ================================================================================================

    std::cout << "Stop the robot " << std::endl;
    robot.setRobotState(vpRobot::STATE_STOP);

    teleop.stop();

    if (opt_plot && plotter != nullptr) {
      delete plotter;
      plotter = nullptr;
    }

    if (traj_corners) {
      delete[] traj_corners;
    }
  }
  catch (const vpException &e) {
    std::cout << "ViSP exception: " << e.what() << std::endl;
    std::cout << "Stop the robot " << std::endl;
    robot.setRobotState(vpRobot::STATE_STOP);
    teleop.stop();

#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
    if (display != nullptr) {
      delete display;
    }
#endif
    return EXIT_FAILURE;
  }
  catch (const std::exception &e) {
    std::cout << "std::exception: " << e.what() << std::endl;
    teleop.stop();
#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
    if (display != nullptr) {
      delete display;
    }
#endif
    return EXIT_FAILURE;
  }

#if (VISP_CXX_STANDARD < VISP_CXX_STANDARD_11)
  if (display != nullptr) {
    delete display;
  }
#endif
  return EXIT_SUCCESS;
}


#else
int main()
{
#if !defined(VISP_HAVE_REALSENSE2)
  std::cout << "Install librealsense-2.x" << std::endl;
#endif
#if !defined(VISP_HAVE_UR_RTDE)
  std::cout << "ViSP is not built with libur_rtde 3rd party used to control a robot from Universal Robots..."
            << std::endl;
#endif
  return EXIT_SUCCESS;
}
#endif
