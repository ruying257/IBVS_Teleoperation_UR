#include "RobotTeleoperation.h"

#include <iostream>
#include <unistd.h>
#include <fcntl.h>
#include <locale.h>
#include <csignal>

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

// ControlVector 类实现
RobotTeleoperation::ControlVector::ControlVector() 
    : pose_deltas(6, 0.0), joint_deltas(6, 0.0) {}

void RobotTeleoperation::ControlVector::zero() {
    std::fill(pose_deltas.begin(), pose_deltas.end(), 0.0);
    std::fill(joint_deltas.begin(), joint_deltas.end(), 0.0);
}

// KeyState 类实现
void RobotTeleoperation::KeyState::reset_all() {
    w = x = s = a = d = r = f = false;
    i = k = j = l = u = o = false;
    num1 = num2 = num3 = num4 = num5 = num6 = false;
    num1_shift = num2_shift = num3_shift = num4_shift = num5_shift = num6_shift = false;
    fine = exit_flag = estop = false;
}

void RobotTeleoperation::KeyState::reset() {
    w = x = s = a = d = r = f = false;
    i = k = j = l = u = o = false;
    num1 = num2 = num3 = num4 = num5 = num6 = false;
    num1_shift = num2_shift = num3_shift = num4_shift = num5_shift = num6_shift = false;
}

// RobotTeleoperation 类实现
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

bool RobotTeleoperation::isRunning() const {
    return running;
}
