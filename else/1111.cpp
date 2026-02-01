#include "tensorrt_ibvs_3.h"
#include "admittance.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

#include <time.h>


TensorRT_ibvs_3::TensorRT_ibvs_3(QObject *parent)
    : QThread(parent), m_running(false), m_robot(nullptr), m_rs(nullptr), boltDetector(nullptr)
{
    boltDetector = new TensorRT_detection("/home/z/RemoteControl-v4/model/best.trt");
}

TensorRT_ibvs_3::~TensorRT_ibvs_3() {
    stop();
    if (m_robot) delete m_robot;
    if (m_rs) delete m_rs;
}

void TensorRT_ibvs_3::stop() {
    m_running = false;
    // 停止机械臂
    if (m_robot) {
        try {
            m_robot->setRobotState(vpRobot::STATE_STOP);
        } catch (const vpException &e) {
            qDebug() << "停止机械臂时发生错误:" << QString::fromStdString(e.getStringMessage());
        }
    }
    wait(1000); // 等待线程结束
}

void TensorRT_ibvs_3::display_point_trajectory(const vpImage<vpRGBa> &I, const std::vector<vpImagePoint> &vip,
                                             std::vector<vpImagePoint> *traj_vip)
{
    for (size_t i = 0; i < vip.size(); i++) {
        if (traj_vip[i].size()) {
            // Add the point only if distance with the previous > 1 pixel
            if (vpImagePoint::distance(vip[i], traj_vip[i].back()) > 1.) {
                traj_vip[i].push_back(vip[i]);
            }
        } else {
            traj_vip[i].push_back(vip[i]);
        }
    }
    for (size_t i = 0; i < vip.size(); i++) {
        for (size_t j = 1; j < traj_vip[i].size(); j++) {
            vpDisplay::displayLine(I, traj_vip[i][j - 1], traj_vip[i][j], vpColor::green, 2);
        }
    }
}


TensorRT_ibvs_3::PixelROI TensorRT_ibvs_3::computeBoltPlaneROI(
    const std::vector<vpImagePoint>& boltPoints,
    int imgWidth,
    int imgHeight,
    double minSize,
    double paddingRatio
) {
    if (boltPoints.size() < 2) {
        throw std::runtime_error("螺栓点数量不足，无法计算 ROI");
    }

    double u_min = boltPoints[0].get_u();
    double u_max = u_min;
    double v_min = boltPoints[0].get_v();
    double v_max = v_min;

    for (size_t i = 1; i < boltPoints.size(); ++i) {
        double u = boltPoints[i].get_u();
        double v = boltPoints[i].get_v();

        u_min = std::min(u_min, u);
        u_max = std::max(u_max, u);
        v_min = std::min(v_min, v);
        v_max = std::max(v_max, v);
    }

    double roi_w = std::max(u_max - u_min, minSize);
    double roi_h = std::max(v_max - v_min, minSize);
    double pad_x = paddingRatio * roi_w;
    double pad_y = paddingRatio * roi_h;

    PixelROI roi;
    roi.x0 = static_cast<int>(u_min - pad_x);
    roi.y0 = static_cast<int>(v_min - pad_y);
    roi.x1 = static_cast<int>(u_max + pad_x);
    roi.y1 = static_cast<int>(v_max + pad_y);

    roi.x0 = std::max(0, roi.x0);
    roi.y0 = std::max(0, roi.y0);
    roi.x1 = std::min(imgWidth  - 1, roi.x1);
    roi.y1 = std::min(imgHeight - 1, roi.y1);

    if (roi.x1 <= roi.x0 || roi.y1 <= roi.y0) {
        throw std::runtime_error("ROI 计算失败（非法区域）");
    }

    return roi;
}


vpHomogeneousMatrix TensorRT_ibvs_3::estimateBoltPlanePose(
    const std::vector<vpImagePoint>& m_finalBoltPoints,
    const vpImage<uint16_t>& depth,
    int WIDTH,
    int HEIGHT,
    double fx, double fy, double cx, double cy,
    double depth_scale
) {
    // 0. ROI 计算
    PixelROI roi = computeBoltPlaneROI(m_finalBoltPoints, WIDTH, HEIGHT);
    int x0 = roi.x0, y0 = roi.y0;
    int x1 = roi.x1, y1 = roi.y1;

    // 1. ROI → 相机坐标系点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cam(new pcl::PointCloud<pcl::PointXYZ>);
    for (int v = y0; v < y1; ++v) {
        for (int u = x0; u < x1; ++u) {
            double z_raw = depth[u][v];
            if (z_raw <= 0) continue;

            double z = z_raw / depth_scale;
            if (z < 0.05 || z > 5.0) continue;

            double X = (u - cx) * z / fx;
            double Y = (v - cy) * z / fy;

            cloud_cam->points.emplace_back(X, Y, z);
        }
     }

    cloud_cam->width  = cloud_cam->points.size();
    cloud_cam->height = 1;
    if (cloud_cam->points.size() < 200) {
        throw std::runtime_error("ROI 内点数过少，无法拟合平面");
    }

    // 2. RANSAC 平面拟合
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(0.001);
    seg.setMaxIterations(1000);

    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coeffs(new pcl::ModelCoefficients);

    seg.setInputCloud(cloud_cam);
    seg.segment(*inliers, *coeffs);
    if (inliers->indices.empty()) {
        throw std::runtime_error("平面拟合失败");
    }

    // 3. 平面法向量（相机系）+ 方向统一
    vpColVector normal_cam(3);
    normal_cam[0] = coeffs->values[0];
    normal_cam[1] = coeffs->values[1];
    normal_cam[2] = coeffs->values[2];
    normal_cam.normalize();

    vpColVector cam_Z(3);
    cam_Z[0] = 0; cam_Z[1] = 0; cam_Z[2] = 1;
    if (vpColVector::dotProd(normal_cam, cam_Z) > 0) {
        normal_cam = -normal_cam;
    }

    // 4. 平面中心点
    vpColVector center_cam(3);
    center_cam = 0.0;
    for (int idx : inliers->indices) {
        center_cam[0] += cloud_cam->points[idx].x;
        center_cam[1] += cloud_cam->points[idx].y;
        center_cam[2] += cloud_cam->points[idx].z;
    }
    center_cam /= static_cast<double>(inliers->indices.size());

    // 5. 平面局部坐标系
    vpColVector z_axis = normal_cam;

    vpColVector ref(3);
    ref[0] = 0; ref[1] = 1; ref[2] = 0;

    vpColVector x_axis = vpColVector::crossProd(ref, z_axis);
    if (x_axis.euclideanNorm() < 1e-6) {
        ref[0] = 1; ref[1] = 0;
        x_axis = vpColVector::crossProd(ref, z_axis);
    }
    x_axis.normalize();

    vpColVector y_axis = vpColVector::crossProd(z_axis, x_axis);
    y_axis.normalize();

    vpRotationMatrix plane_R_cam;
    for (int i = 0; i < 3; ++i) {
        plane_R_cam[i][0] = x_axis[i];
        plane_R_cam[i][1] = y_axis[i];
        plane_R_cam[i][2] = z_axis[i];
    }

//    plane_R_cam[0][0] = 1;
//    plane_R_cam[0][1] = 0;
//    plane_R_cam[0][2] = 0;

//    plane_R_cam[1][0] = 0;
//    plane_R_cam[1][1] = 1;
//    plane_R_cam[1][2] = 0;

//    plane_R_cam[2][0] = 0;
//    plane_R_cam[2][1] = 0;
//    plane_R_cam[2][2] = 1;

//    // 5. 平面局部坐标系（Y轴为法向量）
//    vpColVector y_axis = normal_cam;  // Y 轴作为法向量

//    vpColVector ref(3);
//    ref[0] = 0; ref[1] = 0; ref[2] = 1;
//    vpColVector x_axis = vpColVector::crossProd(y_axis, ref);
//    if (x_axis.euclideanNorm() < 1e-6) {
//        ref[0] = 1; ref[1] = 0; ref[2] = 0;
//        x_axis = vpColVector::crossProd(y_axis, ref);
//    }
//    x_axis.normalize();
//    vpColVector z_axis = vpColVector::crossProd(x_axis, y_axis);
//    z_axis.normalize();

//    vpRotationMatrix plane_R_cam;
//    for (int i = 0; i < 3; ++i) {
//        plane_R_cam[i][0] = x_axis[i];
//        plane_R_cam[i][1] = y_axis[i];
//        plane_R_cam[i][2] = z_axis[i];
//    }

//    std::cout << "============> " << center_cam[0] << " " << center_cam[1] << " " << center_cam[2] << std::endl;

//    vpTranslationVector plane_t(center_cam[0], center_cam[1], center_cam[2]);
    vpTranslationVector plane_t(0, 0, 0);
    vpHomogeneousMatrix plane_targetPose(plane_t, plane_R_cam);
    return plane_targetPose;
}


float TensorRT_ibvs_3::getMedianDepth(const vpImage<uint16_t> &depth_map,  int row, int col, int window_size) {
    std::vector<float> valid_depths;
    int half_win = window_size / 2;

    // 遍历以 (row, col) 为中心的 N*N 窗口
    for (int i = -half_win; i <= half_win; ++i) {
        for (int j = -half_win; j <= half_win; ++j) {
            int r = row + i;
            int c = col + j;

            // 1. 边界安全检查
            if (r >= 0 && r < (int)depth_map.getHeight() && c >= 0 && c < (int)depth_map.getWidth()) {
                uint16_t d = depth_map[r][c];

                // 2. 过滤无效值：RealSense 0 表示无法测距，低于 5mm 通常是噪声
                if (d > 5) {
                    valid_depths.push_back(static_cast<float>(d) / 1000.0f); // 转换为米
                }
            }
        }
    }

    // 3. 处理结果
    if (valid_depths.empty()) {
        return 0.5f; // 如果窗口内全是无效值，返回一个保守的默认距离（50cm）
    }

    // 4. 排序并取中值
    std::sort(valid_depths.begin(), valid_depths.end());
    return valid_depths[valid_depths.size() / 2];
}


void TensorRT_ibvs_3::run() {
    try {
        m_running = true;
        bool opt_verbose = true;
        bool opt_plot = true;
        bool opt_adaptive_gain = false;
        bool opt_task_sequencing = false;
        double convergence_threshold = 0.0001;
        bool final_quit = false;
        bool has_converged = false;
        bool send_velocities = false;
        bool servo_started = false;
        std::vector<vpImagePoint> *traj_corners = nullptr;

        // 初始化机器人
        m_robot = new vpRobotUniversalRobots;
        m_robot->connect(m_robotIP);
        emit servoStatusChanged("Robot connected");

        // 初始化相机参数
        rs2::config config;
        unsigned int width = 640, height = 480;
        config.enable_stream(RS2_STREAM_COLOR,    width, height, RS2_FORMAT_RGBA8, 30);
        config.enable_stream(RS2_STREAM_DEPTH,    width, height, RS2_FORMAT_Z16,   30);
        config.enable_stream(RS2_STREAM_INFRARED, width, height, RS2_FORMAT_Y8,    30);

        m_rs = new vpRealSense2;
        m_rs->open(config);

        rs2::align align_to(RS2_STREAM_COLOR);

        vpImage<vpRGBa> color(
            static_cast<unsigned int>(m_rs->getIntrinsics(RS2_STREAM_COLOR).height),
            static_cast<unsigned int>(m_rs->getIntrinsics(RS2_STREAM_COLOR).width)
        );
        vpImage<vpRGBa> depth_display(
            static_cast<unsigned int>(m_rs->getIntrinsics(RS2_STREAM_DEPTH).height),
            static_cast<unsigned int>(m_rs->getIntrinsics(RS2_STREAM_DEPTH).width)
        );
        vpImage<unsigned char> infrared(
            static_cast<unsigned int>(m_rs->getIntrinsics(RS2_STREAM_INFRARED).height),
            static_cast<unsigned int>(m_rs->getIntrinsics(RS2_STREAM_INFRARED).width)
        );
        vpImage<uint16_t> depth(depth_display.getHeight(), depth_display.getWidth());

        // 初始化相机后创建显示窗口
        vpDisplayX display_color;
        m_rs->acquire(color);
        display_color.init(color, 100, 100, "Color Image");
        vpDisplayX display_depth;
        display_depth.init(depth_display, 100, static_cast<int>(color.getHeight()) + 70, "Depth Image");
        emit servoStatusChanged("Camera initialized");

        vpCameraParameters cam = m_rs->getCameraParameters(RS2_STREAM_COLOR, vpCameraParameters::perspectiveProjWithDistortion);
        std::cout << "============> Camera Parameters: " << cam <<std::endl;

        // 相机相对于机械臂末端的外参
        vpPoseVector ePc;
        if (!m_eMcFilename.empty()) {
            ePc.loadYAML(m_eMcFilename, ePc);
        } else {
            // Realsense D435
//            ePc[0] = 0.118694; ePc[1] = 0.0249475; ePc[2] = -0.0719342;
//            ePc[3] = -1.79448; ePc[4] = 0.762252;  ePc[5] = 1.73034;

//            // Realsense D405
            ePc[0] = 0.141335; ePc[1] = 0.174283; ePc[2] = 0.0915465;
            ePc[3] = -0.556397; ePc[4] = 1.79451;  ePc[5] = -1.77789;
        }
        vpHomogeneousMatrix eMc(ePc);
        m_robot->set_eMc(eMc);

        // 移动到安全位置
        vpColVector q(6, 0);
//        q[0] = vpMath::rad(166.69);  q[1] = vpMath::rad(-101.85); q[2] = vpMath::rad(104.16);
//        q[3] = vpMath::rad(-105.94); q[4] = vpMath::rad(89.03);   q[5] = vpMath::rad(67.06);

//        q[0] = vpMath::rad(162.83);  q[1] = vpMath::rad(-83.29);  q[2] = vpMath::rad(82.44);
//        q[3] = vpMath::rad(-100.4);  q[4] = vpMath::rad(94.44);   q[5] = vpMath::rad(49.6);

//        q[0] = vpMath::rad(22.12);  q[1] = vpMath::rad(-185.73);  q[2] = vpMath::rad(108.14);
//        q[3] = vpMath::rad(5.49);   q[4] = vpMath::rad(96.11);     q[5] = vpMath::rad(34.54);

        q[0] =  vpMath::rad(152.65);
        q[1] = -vpMath::rad(110.89);
        q[2] =  vpMath::rad(119.46);
        q[3] = -vpMath::rad(103.64);
        q[4] =  vpMath::rad(90.38);
        q[5] = -vpMath::rad(107.6);

        m_robot->setRobotState(vpRobot::STATE_POSITION_CONTROL);
        m_robot->setPosition(vpRobot::JOINT_STATE, q);
        emit servoStatusChanged("Moved to safe position");

        // 初始化视觉伺服任务
        vpServo task;
        std::vector<vpFeaturePoint> p(4), pd(4);
        for (size_t i = 0; i < p.size(); i++) {
            task.addFeature(p[i], pd[i]);
        }
        task.setServo(vpServo::EYEINHAND_CAMERA);
        task.setInteractionMatrixType(vpServo::CURRENT);
        task.setLambda(0.5);

        // 设置目标特征点
        std::vector<vpPoint> point(4);

        double scale = 1.0;
        point[0].setWorldCoordinates( 0.04911473583  * scale,  0.03522975298 * scale, 0);
        point[1].setWorldCoordinates(-0.004974177597 * scale,  0.03811132624 * scale, 0);
        point[2].setWorldCoordinates(-0.00723769331  * scale, -0.01442762387 * scale, 0);
        point[3].setWorldCoordinates( 0.04751209846  * scale, -0.01489772015 * scale, 0);

        // 位姿矩阵
        vpHomogeneousMatrix cdMc, cMo, oMo;
        vpHomogeneousMatrix cdMo[4];
        cdMo[0].buildFrom(vpTranslationVector(0, 0, 0.3759999871), vpRotationMatrix({1, 0, 0, 0, -1, 0, 0, 0, -1}));
        cdMo[1].buildFrom(vpTranslationVector(0, 0, 0.3770000041), vpRotationMatrix({1, 0, 0, 0, -1, 0, 0, 0, -1}));
        cdMo[2].buildFrom(vpTranslationVector(0, 0, 0.3819999993), vpRotationMatrix({1, 0, 0, 0, -1, 0, 0, 0, -1}));
        cdMo[3].buildFrom(vpTranslationVector(0, 0, 0.3779999912), vpRotationMatrix({1, 0, 0, 0, -1, 0, 0, 0, -1}));

        // 计算期望特征点的图像坐标
        qDebug() << "cP: ";
        for (size_t i = 0; i < point.size(); i++) {
            vpColVector cP, p_;
            point[i].changeFrame(cdMo[i], cP);
            point[i].projection(cP, p_);
            pd[i].set_x(p_[0]);
            pd[i].set_y(p_[1]);
            pd[i].set_Z(cP[2]);
            qDebug()<< i << ": " << cP[0] << ", "<< cP[1] << ", " << cP[2];
        }

        qDebug() << "pd: " ;
        for (size_t i = 0; i < point.size(); i++) {
            qDebug() << i << ": " << pd[i].get_x() << ", "<< pd[i].get_y();
        }

        // 绘图参数设置
        vpPlot *plotter = nullptr;
        int iter_plot = 0;
        if (opt_plot) {
            plotter = new vpPlot(4, static_cast<int>(250 * 4), 1000, static_cast<int>(color.getWidth()) + 80, 10, "Real time curves plotter");
            plotter->setTitle(0, "Visual features error");
            plotter->setTitle(1, "Camera velocities");
            plotter->setTitle(2, "joint positions");
            plotter->setTitle(3, "joint velocitys");
            plotter->initGraph(0, 8);
            plotter->initGraph(1, 6);
            plotter->initGraph(2, 6);
            plotter->initGraph(3, 6);
            plotter->setLegend(0, 0, "e1_x");
            plotter->setLegend(0, 1, "e1_y");
            plotter->setLegend(0, 2, "e2_x");
            plotter->setLegend(0, 3, "e2_y");
            plotter->setLegend(0, 4, "e3_x");
            plotter->setLegend(0, 5, "e3_y");
            plotter->setLegend(0, 6, "e4_x");
            plotter->setLegend(0, 7, "e4_y");
            plotter->setLegend(1, 0, "vc_x");
            plotter->setLegend(1, 1, "vc_y");
            plotter->setLegend(1, 2, "vc_z");
            plotter->setLegend(1, 3, "wc_x");
            plotter->setLegend(1, 4, "wc_y");
            plotter->setLegend(1, 5, "wc_z");
            plotter->setLegend(2, 0, "jp1");
            plotter->setLegend(2, 1, "jp2");
            plotter->setLegend(2, 2, "jp3");
            plotter->setLegend(2, 3, "jp4");
            plotter->setLegend(2, 4, "jp5");
            plotter->setLegend(2, 5, "jp6");
        }

        static double t_init_servo = vpTime::measureTimeMs();
        m_robot->setRobotState(vpRobot::STATE_VELOCITY_CONTROL);
        emit servoStatusChanged("Servo started");

        auto receive = m_robot->getRTDEReceiveInterfaceHandler();
        auto control = m_robot->getRTDEControlInterfaceHandler();


        double fx_ = cam.get_px();  // 焦距 x
        double fy_ = cam.get_py();  // 焦距 y
        double cx_ = cam.get_u0();  // 主点 x
        double cy_ = cam.get_v0();  // 主点 y
        double kud = cam.get_kud();
        double kdu = cam.get_kdu();


        std::cout << "fx: " << fx_ << " fy: " << fy_ << "cx: " << cx_ << " cy: " << cy_ << " kud: " << kud << " kdu: " << kdu;

        while (m_running && !has_converged && !final_quit) {
            double t_start = vpTime::measureTimeMs();
            std::vector<vpImagePoint> boltPoints;

            // 获取相机图像
            m_rs->acquire(reinterpret_cast<unsigned char *>(color.bitmap), reinterpret_cast<unsigned char *>(depth.bitmap), nullptr, nullptr, &align_to);

            // **********************************************检测螺栓****************************************************
            FrameData frame;
            frame.image = color;
            frame.timestamp = QDateTime::currentMSecsSinceEpoch();

            DetectionResult result;
            boltDetector->infer_trtmodel(frame, result);
            for (const auto& bolt : result.bolts) {
                // 获取边界框坐标
                vpRect bbox = bolt.bounding_box;

                // 计算边界框中心点
                double center_x = bbox.getLeft() + bbox.getWidth() / 2.0;
                double center_y = bbox.getTop() + bbox.getHeight() / 2.0;
                vpImagePoint center(center_y, center_x);  // vpImagePoint的构造函数是(y, x)
                boltPoints.push_back(center);

                vpDisplay::displayRectangle(
                    color,
                    bbox,
                    vpColor::red,  // 框颜色
                    false,         // 不填充
                    2              // 线宽
                );

                // 在框上方添加文本（类别 + 置信度）
                std::ostringstream oss;
                oss << bolt.class_name << " " << std::fixed << std::setprecision(2) << bolt.confidence;
                vpDisplay::displayText(
                    color,
                    static_cast<int>(bbox.getTop() - 15),
                    static_cast<int>(bbox.getLeft()),
                    oss.str(),
                    vpColor::green
                );
            }

            // 计算四个螺栓点的中心（用于方向角计算）
            vpImagePoint centroid(0.0, 0.0);
            for (const auto& pt : boltPoints) {
                centroid += pt;
            }
            centroid /= static_cast<double>(boltPoints.size());
            std::vector<BoltAngleData> angleData;
            angleData.reserve(boltPoints.size());

            // 计算每个点相对于中心的方向角
            for (size_t i = 0; i < boltPoints.size(); i++) {
                double dx = boltPoints[i].get_u() - centroid.get_u();  // x 方向偏移
                double dy = boltPoints[i].get_v() - centroid.get_v();  // y 方向偏移
                double angle_rad = std::atan2(dy, dx);
                if (angle_rad < 0) {
                    angle_rad += 2 * M_PI;
                }

                angleData.push_back({i, boltPoints[i], angle_rad});
            }

            // 按角度排序（从小到大，即逆时针顺序）
            std::sort(angleData.begin(), angleData.end(), [](const BoltAngleData& a, const BoltAngleData& b) { return a.angle > b.angle; });
            std::vector<vpImagePoint> sortedPoints;
            for (const auto& data : angleData) {
                sortedPoints.push_back(data.point);
            }
            boltPoints = sortedPoints;

            // 显示图像
            vpImageConvert::createDepthHistogram(depth, depth_display);
            vpDisplay::display(color);
            vpDisplay::display(depth_display);
            vpDisplay::flush(color);
            vpDisplay::flush(depth_display);
            {
                std::stringstream ss;
                ss << "Left click to " << (send_velocities ? "stop the robot" : "servo the robot") << ", right click to quit.";
                vpDisplay::displayText(color, 20, 20, ss.str(), vpColor::red);
            }
            for (size_t i = 0; i < pd.size(); i++) {
                std::stringstream ss; ss << i;

                vpImagePoint ip;
                vpMeterPixelConversion::convertPoint(cam, pd[i].get_x(), pd[i].get_y(), ip);  // 显示期望螺栓点索引
                vpDisplay::displayText(color, ip + vpImagePoint(15, 15), ss.str(), vpColor::red);
            }

            vpColVector v_c(6); 
            if(boltDetector && boltPoints.size() == 4){
                static bool first_time = true;
                // 计算当前图像的螺栓四个中心点的2D特征坐标（x，y）
                for (size_t i = 0; i < boltPoints.size(); i++) {
                    // 创建特征，生成当前特征点的2D特征坐标（x，y）
                    vpFeatureBuilder::create(p[i], cam, boltPoints[i]);

                    // 获取螺栓中心点坐标
                    vpImagePoint center = boltPoints[i];

                    // 将浮点坐标转换为整数索引（确保在图像范围内）
                    unsigned int row = static_cast<unsigned int>(std::min(std::max(center.get_i(), 0.0), static_cast<double>(depth_display.getHeight() - 1)));
                    unsigned int col = static_cast<unsigned int>(std::min(std::max(center.get_j(), 0.0), static_cast<double>(depth_display.getWidth() - 1)));

                    // 从深度图获取深度值
                    float depth_value = 0.0f;
                    if (depth.getSize() > 0 && depth[row][col] > 0.001) {
                        // D435 相机
//                        depth_value = static_cast<float>(depth[row][col]) / 1000.0f;  // 距离单位转换为米

                        // D405相机
                        depth_value = static_cast<float>(depth[row][col]) / 10000.0f;  // 距离单位转换为米
                    } else {
                        depth_value = 0.1f;  // 安全默认值
                    }

                    p[i].set_Z(depth_value);
                }

                // 保存最后一帧数据的螺栓坐标信息
                m_finalBoltPoints = boltPoints;
                m_finalBoltDepths.clear();
                for (size_t i = 0; i < 4; ++i) {
                   m_finalBoltDepths.push_back(p[i].get_Z());
                }
                finalDepth = depth;

                // 选择控制律的计算模式
                if (opt_task_sequencing) {                                      // 分阶段任务控制模式（时间依赖的控制策略和标准控制模式）
                    if (!servo_started) {
                        if (send_velocities) {                                  // 左键点击后 send_velocities 转换，开启视觉伺服
                            servo_started = true;
                        }
                        t_init_servo = vpTime::measureTimeMs();                 // 记录左键点击视觉伺服启动的启动时间
                    }
                    v_c = task.computeControlLaw((vpTime::measureTimeMs() - t_init_servo) / 1000.);
                } else {
                    v_c = task.computeControlLaw();
                }
                vpColVector q;  // 创建存储关节角度的向量
                m_robot->getPosition(vpRobot::JOINT_STATE, q);

                // 图像窗口中当前特征点的标注和特征点轨迹绘制
                vpServoDisplay::display(task, cam, color);
                for (size_t i = 0; i < boltPoints.size(); i++) {
                    std::stringstream ss; ss << i;
                    vpDisplay::displayText(color, boltPoints[i] + vpImagePoint(15, 15), ss.str(), vpColor::green);
                }

                if (first_time) {
                    traj_corners = new std::vector<vpImagePoint>[boltPoints.size()];
                }
//                 display_point_trajectory(color, boltPoints, traj_corners);

                if (opt_plot) {
                    double t = (vpTime::measureTimeMs() - t_init_servo)/1000.0;
                    plotter->plot(0, t, task.getError());
                    plotter->plot(1, t, v_c);
                    plotter->plot(2, t, q);
                    iter_plot++;

                    // 保存数据到文件
                    if (data_file.is_open()) {
                        data_file << std::fixed << std::setprecision(6) << t << ",";

                        // 写入误差向量
                        vpColVector error = task.getError();
                        for (unsigned i = 0; i < error.size(); i++) {
                            data_file << error[i] << ",";
                        }

                        // 写入控制速度
                        for (int i = 0; i < 6; i++) {
                            data_file << v_c[i] << ",";
                        }

                        // 写入关节角度
                        for (int i = 0; i < 6; i++) {
                            data_file << q[i];
                            if (i < 5) data_file << ",";
                        }
                        data_file << "\n";
                    }
                }

                // 特征点误差曲线图绘制
                double error = task.getError().sumSquare();
                std::stringstream ss;
                ss << "error: " << error;
                vpDisplay::displayText(color, 20, static_cast<int>(color.getWidth()) - 150, ss.str(), vpColor::red);

                if (opt_verbose)
                    if (error < convergence_threshold) {
                        has_converged = true;

                        // 保存关节角度
                        vpColVector q_end(6);
                        m_robot->getPosition(vpRobot::JOINT_STATE, q_end);
                        m_convergenceJointAngles = q_end;

                        std::cout << "Servo task has converged" << "\n";
                        vpDisplay::displayText(color, 100, 20, "Servo task has converged", vpColor::red);

                        m_robot->setVelocity(vpRobot::CAMERA_FRAME, {0, 0, 0, 0, 0, 0});
                    }

                if (first_time) {
                    first_time = false;
                }
            } else {
                v_c = 0;
                vpDisplay::displayText(color, 20, 400, "No bolts detected or not enough bolts", vpColor::red);
            }


            if (!send_velocities) {
                v_c = 0;
            }

            // 将相机速度指令发送给机械臂，核心控制指令
            m_robot->setVelocity(vpRobot::CAMERA_FRAME, v_c);

            // GUI界面数据更新
            try {
                auto pose = receive ? receive->getActualTCPPose() : std::vector<double>();
                auto joints = receive ? receive->getActualQ() : std::vector<double>();
                auto joints_current = receive ? receive->getActualCurrent() : std::vector<double>();
                auto joints_torque = receive ? receive->getTargetMoment() : std::vector<double>();
                auto ActualTCPForce = receive ? receive->getActualTCPForce(): std::vector<double>();
                if (!pose.empty()) {
                    emit ibvs_poseUpdated(
                             pose[0], pose[1], pose[2],
                             pose[3], pose[4], pose[5],
                             joints[0], joints[1],joints[2],
                             joints[3], joints[4],joints[5],
                             joints_current[0], joints_current[1],joints_current[2],
                             joints_current[3], joints_current[4],joints_current[5],
                             joints_torque[0], joints_torque[1],joints_torque[2],
                             joints_torque[3], joints_torque[4],joints_torque[5],
                             ActualTCPForce[0],ActualTCPForce[1],ActualTCPForce[2],
                             ActualTCPForce[3],ActualTCPForce[4],ActualTCPForce[5]
                         );
                }
            } catch (const std::exception& e) {
                emit errorOccurred(QString("获取位姿失败: %1").arg(e.what()));
                break;
            }

            // *****************************图像窗口提示、图像刷新、鼠标左右按键设置***********************************************
            {
                std::stringstream ss,ee;
                ss << "Loop time: " << vpTime::measureTimeMs() - t_start << " ms";
                ee << "FPS: " << 1000/(vpTime::measureTimeMs() - t_start) << " hz";
                vpDisplay::displayText(color, 40, 20, ss.str(), vpColor::red);
                vpDisplay::displayText(color, 60, 20, ee.str(), vpColor::red);

            }
            vpDisplay::flush(color);

            vpMouseButton::vpMouseButtonType button;
            if (vpDisplay::getClick(color, button, false)) {
                switch (button) {
                case vpMouseButton::button1:
                    send_velocities = !send_velocities;
                    break;
                case vpMouseButton::button3:
                    final_quit = true;
                    v_c = 0;
                    break;
                default:
                    break;
                }
            }

            if (!m_running) break;
        }

       // 第二段：视觉引导
       m_robot->setRobotState(vpRobot::STATE_POSITION_CONTROL);

       double fx = cam.get_px();  // 焦距 x
       double fy = cam.get_py();  // 焦距 y
       double cx = cam.get_u0();  // 主点 x
       double cy = cam.get_v0();  // 主点 y
       double depth_scale = 10000.0;

       std::cout << "fx: " << fx << " fy: " << fy << "cx: " << cx << " cy: " << cy;

//       double tcpX = -98.33 / 1000;      // 控制左右（绝对值数值越大，套筒偏右）
//       double tcpY =  87.61 / 1000;      // 控制前后（绝对值数值越大远离螺栓）
//       double tcpZ = 189.58 / 1000;      // 控制上下（绝对值数值越大，套筒偏下），188.5
//       double rpyRx = 0;
//       double rpyRy = 0;
//       double rpyRz = 0;

//       vpRotationMatrix eMt_R;
//       eMt_R.buildFrom(vpRxyzVector(rpyRx, rpyRy, rpyRz));
//       vpTranslationVector eMt_t(tcpX, tcpY, tcpZ);
//       vpHomogeneousMatrix eMt(eMt_t, eMt_R);

//       vpHomogeneousMatrix robot_bMt = m_robot->get_fMe();
//       vpHomogeneousMatrix bMc = robot_bMt * eMc;

       vpHomogeneousMatrix plane_targetPose = estimateBoltPlanePose(
           m_finalBoltPoints,
           finalDepth,
           width,
           height,
           fx, fy, cx, cy,
           depth_scale
       );
//       m_robot->setPosition(vpRobot::CAMERA_FRAME, plane_targetPose);

       // 1. 获取 IBVS 收敛后的当前位姿
       vpPoseVector T_c_current;
       m_robot->getPosition(vpRobot::CAMERA_FRAME, T_c_current);

       // 2. 平面法向（相机系）
       vpColVector n_plane_c(3);
       n_plane_c[0] = plane_targetPose[0][2];
       n_plane_c[1] = plane_targetPose[1][2];
       n_plane_c[2] = plane_targetPose[2][2];

       // 3. 相机光轴
       vpColVector z_cam(3);
       z_cam[0] = 0.0;
       z_cam[1] = 0.0;
       z_cam[2] = 1.0;

       // 4. 姿态误差 → 微小旋转
       vpColVector axis = vpColVector::crossProd(n_plane_c, z_cam);
       double sin_theta = axis.euclideanNorm();

       if (sin_theta > 1e-6) {
           axis /= sin_theta;
           double max_angle = vpMath::rad(3.0);
           double theta = std::min(sin_theta, max_angle);

           vpThetaUVector tu;
           tu[0] = axis[0] * theta;
           tu[1] = axis[1] * theta;
           tu[2] = axis[2] * theta;

           vpRotationMatrix R_delta;
           R_delta.buildFrom(tu);

           vpHomogeneousMatrix T_delta;
           T_delta.eye();
           T_delta.insert(R_delta);

           vpHomogeneousMatrix T_c_new = vpHomogeneousMatrix(T_c_current) * T_delta;
           m_robot->setPosition(vpRobot::CAMERA_FRAME, T_c_new);
       }

//       double fx = cam.get_px();  // 焦距 x
//       double fy = cam.get_py();  // 焦距 y
//       double cx = cam.get_u0();  // 主点 x
//       double cy = cam.get_v0();  // 主点 y
//       for (size_t i = 0; i < 4; i++) {
//           double u = m_finalBoltPoints[i].get_u();
//           double v = m_finalBoltPoints[i].get_v();
//           double Z = m_finalBoltDepths[i];
//           double X = (u - cx) * Z / fx;
//           double Y = (v - cy) * Z / fy;

//           vpColVector coord(4);
//           coord[0] = X;
//           coord[1] = Y;
//           coord[2] = Z;
//           coord[3]= 1.0;
//           boltCameraCoords.push_back(coord);
//       }

//       // 相机坐标——>世界坐标
//       //**************************************机械臂末端到基座的变化矩阵bMe***********************************************//
//       std::vector<double> joints = receive->getActualQ();

//       qDebug() << "最后一帧的关节角度:";
//       qDebug() << joints[0] << joints[1] << joints[2] << joints[3] << joints[4] << joints[5];

//       std::vector<double> test_tcp(6);
//       auto test_bMe = control->getForwardKinematics(joints, test_tcp);

//       qDebug() << "最后一帧的位姿向量:";
//       qDebug() << test_bMe[0] << test_bMe[1] << test_bMe[2] << test_bMe[3] << test_bMe[4] << test_bMe[5];

//       vpRzyxVector test_rpy(test_bMe[5], test_bMe[3], test_bMe[4]);
//       vpRotationMatrix test_R_TCP(test_rpy);
//       vpHomogeneousMatrix test_bMe_Matrix;
//       test_bMe_Matrix.insert(test_R_TCP);
//       test_bMe_Matrix[0][3] = test_bMe[0];
//       test_bMe_Matrix[1][3] = test_bMe[1];
//       test_bMe_Matrix[2][3] = test_bMe[2];

////       qDebug() << "最后一帧的基座-末端变换矩阵bMe:";
////       for (int i = 0; i < 4; i++) {
////           qDebug() << test_bMe_Matrix[i][0] << test_bMe_Matrix[i][1] << test_bMe_Matrix[i][2] << test_bMe_Matrix[i][3];
////       }

//       //*************************************工具端到基座的变化矩阵bMt***************************************************//
//       vpHomogeneousMatrix robot_bMt = m_robot->get_fMe();
////       qDebug() << "bMt:";
////       for (int i = 0; i < 4; i++) {
////           qDebug() << robot_bMt[i][0] << robot_bMt[i][1] << robot_bMt[i][2] << robot_bMt[i][3];
////       }

//       //*************************************相机到基座的变化矩阵bMc*****************************************************//
//       vpHomogeneousMatrix bMc = robot_bMt * eMc;
////       qDebug() << "最后一帧的基座-相机变换矩阵bMc:";
////       for (int i = 0; i < 4; i++) {
////           qDebug() << bMc[i][0] << bMc[i][1] << bMc[i][2] << bMc[i][3];
////       }

//       //*************************************计算螺栓的世界坐标**********************************************************//
//       for (size_t i = 0; i < boltCameraCoords.size(); ++i) {
//           vpColVector P_b = bMc * boltCameraCoords[i];

//           vpColVector world_coord(3);
//           world_coord[0] = P_b[0];  // Xw
//           world_coord[1] = P_b[1];  // Yw
//           world_coord[2] = P_b[2];  // Zw
//           boltWorldCoords.push_back(world_coord);
//           qDebug() << "螺栓" << i << "世界坐标："<< world_coord[0] << "m, " << world_coord[1] << "m, " << world_coord[2] << "m";
//       }

//       //*************************************计算螺栓的位姿*************************************************************//
//       // 1. 计算螺栓平面的法向量。这将是目标位姿的【Y轴】
//       vpColVector vec_temp1 = boltWorldCoords[1] - boltWorldCoords[0];  // P0 -> P1 (左)
//       vpColVector vec_temp2 = boltWorldCoords[3] - boltWorldCoords[0];  // P0 -> P3 (下)
//       vpColVector new_Y_axis = vpColVector::crossProd(vec_temp1, vec_temp2);
//       new_Y_axis.normalize();
//       // (方向检查) 确保 Y 轴（接近轴）是从线夹表面朝外指向机器人的
//       vpColVector camera_Z_in_world(3);
//       camera_Z_in_world[0] = bMc[0][2];
//       camera_Z_in_world[1] = bMc[1][2];
//       camera_Z_in_world[2] = bMc[2][2];
////       if (vpColVector::dotProd(new_Y_axis, camera_Z_in_world) < 0) {
//       new_Y_axis = -new_Y_axis;
////       }

//       // 2. 定义 X 轴
//       // X 轴是从螺栓(P2)指向螺栓(P1)的向量
//       vpColVector new_X_axis = boltWorldCoords[1] - boltWorldCoords[2];
//       new_X_axis.normalize();

//       // 3. 计算最终的 Z 轴
//       vpColVector new_Z_axis = vpColVector::crossProd(new_X_axis, new_Y_axis);
//       new_Z_axis.normalize();

//       // 4. 构建最终的旋转矩阵
//       vpRotationMatrix bolt_R;
//       for (int i = 0; i < 3; ++i) {
//           bolt_R[i][0] = new_X_axis[i];  // 第一列是 X 轴
//           bolt_R[i][1] = new_Y_axis[i];  // 第二列是 Y 轴
//           bolt_R[i][2] = new_Z_axis[i];  // 第三列是 Z 轴
//       }

////       qDebug() << "最终的目标姿态 (旋转矩阵 R):";
////       for (int i = 0; i < 3; i++) {
////            qDebug() << bolt_R[i][0] << bolt_R[i][1] << bolt_R[i][2];
////       }

//       vpTranslationVector bolt0_translation(boltWorldCoords[0][0], boltWorldCoords[0][1], boltWorldCoords[0][2]);
//       vpTranslationVector bolt1_translation(boltWorldCoords[1][0], boltWorldCoords[1][1], boltWorldCoords[1][2]);
//       vpTranslationVector bolt2_translation(boltWorldCoords[2][0], boltWorldCoords[2][1], boltWorldCoords[2][2]);
//       vpTranslationVector bolt3_translation(boltWorldCoords[3][0], boltWorldCoords[3][1], boltWorldCoords[3][2]);

//       vpHomogeneousMatrix bolt0_targetPose(bolt0_translation, bolt_R);  // 右上角
//       vpHomogeneousMatrix bolt1_targetPose(bolt1_translation, bolt_R);  // 左上角
//       vpHomogeneousMatrix bolt2_targetPose(bolt2_translation, bolt_R);  // 左下角
//       vpHomogeneousMatrix bolt3_targetPose(bolt3_translation, bolt_R);  // 右下角

//       //***********************************计算机械臂末端法兰需要达到的位姿************************************************//
//       double tcpX = -98.33 / 1000;      // 控制左右（绝对值数值越大，套筒偏右）
//       double tcpY =  87.61 / 1000;      // 控制前后（绝对值数值越大远离螺栓）
//       double tcpZ = 189.58 / 1000;      // 控制上下（绝对值数值越大，套筒偏下），188.5
//       double rpyRx = 0;
//       double rpyRy = 0;
//       double rpyRz = 0;

//       vpRotationMatrix eMt_R;
//       eMt_R.buildFrom(vpRxyzVector(rpyRx, rpyRy, rpyRz));
//       vpTranslationVector eMt_t(tcpX, tcpY, tcpZ);
//       vpHomogeneousMatrix eMt(eMt_t, eMt_R);

//       vpHomogeneousMatrix flange_0target_pose = bolt0_targetPose * eMt.inverse();
//       vpHomogeneousMatrix flange_1target_pose = bolt1_targetPose * eMt.inverse();
//       vpHomogeneousMatrix flange_2target_pose = bolt2_targetPose * eMt.inverse();
//       vpHomogeneousMatrix flange_3target_pose = bolt3_targetPose * eMt.inverse();

//       // 目标位姿放入数组进行循环
//        std::vector<vpHomogeneousMatrix> targetPoses = {
//            flange_0target_pose,
//            flange_1target_pose,
//            flange_2target_pose,
//            flange_3target_pose
//        };

//        try {
//            // 定义四组不同的位置序列
//            std::vector<std::vector<std::vector<double>>> allPositionGroups(4);

//            // 第一组位置序列
//            allPositionGroups[0] = {
//                {vpMath::rad(111.03), vpMath::rad(-121.07), vpMath::rad(128.24), vpMath::rad(-91.97), vpMath::rad(97.61), vpMath::rad(167.50)},
//                {vpMath::rad(117.03), vpMath::rad(-102.59), vpMath::rad(137.20), vpMath::rad(-44.15), vpMath::rad(100.92), vpMath::rad(167.48)},

//                {vpMath::rad(124.31), vpMath::rad(-78.89), vpMath::rad(131.44), vpMath::rad(-53.32), vpMath::rad(110.34), vpMath::rad(184.02)},
//                {vpMath::rad(124.17), vpMath::rad(-76.66), vpMath::rad(130.52), vpMath::rad(-51.74), vpMath::rad(109.35), vpMath::rad(184.02)},
//                {vpMath::rad(124.31), vpMath::rad(-78.89), vpMath::rad(131.44), vpMath::rad(-53.32), vpMath::rad(110.34), vpMath::rad(184.02)},

//                {vpMath::rad(117.03), vpMath::rad(-102.59), vpMath::rad(137.20), vpMath::rad(-44.15), vpMath::rad(100.92), vpMath::rad(167.48)},
//                {vpMath::rad(111.03), vpMath::rad(-121.07), vpMath::rad(128.24), vpMath::rad(-91.97), vpMath::rad(97.61), vpMath::rad(167.50)},
//                {vpMath::rad(152.23), vpMath::rad(-84.77), vpMath::rad(87.13), vpMath::rad(-114.92), vpMath::rad(96.66), vpMath::rad(121.89)}
//            };

//            // 第二组位置序列
//            allPositionGroups[1] = {
//                {vpMath::rad(111.03), vpMath::rad(-121.07), vpMath::rad(128.24), vpMath::rad(-91.97), vpMath::rad(97.61), vpMath::rad(167.50)},
//                {vpMath::rad(117.03), vpMath::rad(-102.59), vpMath::rad(137.20), vpMath::rad(-44.15), vpMath::rad(100.92), vpMath::rad(167.48)},

//                {vpMath::rad(120.98), vpMath::rad(-75.95), vpMath::rad(127.61), vpMath::rad(-51.17), vpMath::rad(103.03), vpMath::rad(177.70)},
//                {vpMath::rad(120.99), vpMath::rad(-73.99), vpMath::rad(128.71), vpMath::rad(-55.36), vpMath::rad(103.02), vpMath::rad(177.70)},
//                {vpMath::rad(120.98), vpMath::rad(-75.95), vpMath::rad(127.61), vpMath::rad(-51.17), vpMath::rad(103.03), vpMath::rad(177.70)},

//                {vpMath::rad(117.03), vpMath::rad(-102.59), vpMath::rad(137.20), vpMath::rad(-44.15), vpMath::rad(100.92), vpMath::rad(167.48)},
//                {vpMath::rad(111.03), vpMath::rad(-121.07), vpMath::rad(128.24), vpMath::rad(-91.97), vpMath::rad(97.61), vpMath::rad(167.50)},
//                {vpMath::rad(152.23), vpMath::rad(-84.77), vpMath::rad(87.13), vpMath::rad(-114.92), vpMath::rad(96.66), vpMath::rad(121.89)}
//            };

//            // 第三组位置序列
//            allPositionGroups[2] = {
//                {vpMath::rad(111.03), vpMath::rad(-121.07), vpMath::rad(128.24), vpMath::rad(-91.97), vpMath::rad(97.61), vpMath::rad(167.50)},
//                {vpMath::rad(117.03), vpMath::rad(-102.59), vpMath::rad(137.20), vpMath::rad(-44.15), vpMath::rad(100.92), vpMath::rad(167.48)},

//                {vpMath::rad(120.08), vpMath::rad(-72.58), vpMath::rad(122.02), vpMath::rad(-47.90), vpMath::rad(104.86), vpMath::rad(177.71)},
//                {vpMath::rad(119.45), vpMath::rad(-71.28), vpMath::rad(124.05), vpMath::rad(-52.96), vpMath::rad(101.53), vpMath::rad(174.24)},
//                {vpMath::rad(120.08), vpMath::rad(-72.58), vpMath::rad(122.02), vpMath::rad(-47.90), vpMath::rad(104.86), vpMath::rad(177.71)},

//                {vpMath::rad(117.03), vpMath::rad(-102.59), vpMath::rad(137.20), vpMath::rad(-44.15), vpMath::rad(100.92), vpMath::rad(167.48)},
//                {vpMath::rad(111.03), vpMath::rad(-121.07), vpMath::rad(128.24), vpMath::rad(-91.97), vpMath::rad(97.61), vpMath::rad(167.50)},
//                {vpMath::rad(152.23), vpMath::rad(-84.77), vpMath::rad(87.13), vpMath::rad(-114.92), vpMath::rad(96.66), vpMath::rad(121.89)}
//            };

//            // 第四组位置序列
//            allPositionGroups[3] = {
//                {vpMath::rad(111.03), vpMath::rad(-121.07), vpMath::rad(128.24), vpMath::rad(-91.97), vpMath::rad(97.61), vpMath::rad(167.50)},
//                {vpMath::rad(117.03), vpMath::rad(-102.59), vpMath::rad(137.20), vpMath::rad(-44.15), vpMath::rad(100.92), vpMath::rad(167.48)},

//                {vpMath::rad(115.76), vpMath::rad(-71.46), vpMath::rad(119.17), vpMath::rad(-46.99), vpMath::rad(98.60), vpMath::rad(180.01)},
//                {vpMath::rad(117.39), vpMath::rad(-68.48), vpMath::rad(119.70), vpMath::rad(-51.13), vpMath::rad(101.29), vpMath::rad(177.53)},
//                {vpMath::rad(115.76), vpMath::rad(-71.46), vpMath::rad(119.17), vpMath::rad(-46.99), vpMath::rad(98.60), vpMath::rad(180.01)},

//                {vpMath::rad(117.03), vpMath::rad(-102.59), vpMath::rad(137.20), vpMath::rad(-44.15), vpMath::rad(100.92), vpMath::rad(167.48)},
//                {vpMath::rad(111.03), vpMath::rad(-121.07), vpMath::rad(128.24), vpMath::rad(-91.97), vpMath::rad(97.61), vpMath::rad(167.50)},
//                {vpMath::rad(152.23), vpMath::rad(-84.77), vpMath::rad(87.13), vpMath::rad(-114.92), vpMath::rad(96.66), vpMath::rad(121.89)}
//            };

//            // 依次移动到四个螺栓位置
//            for (int i = 0; i < targetPoses.size() && m_running; i++) {
//                // 移动到当前螺栓位置
//                m_robot->setPosition(vpRobot::END_EFFECTOR_FRAME, targetPoses[i]);
//                QThread::sleep(5);

//                if (!m_running) break;

//                try {
//                    // 检查i是否在有效范围内
//                    if (i < allPositionGroups.size()) {
//                        // 获取当前组的位置序列
//                        std::vector<std::vector<double>>& currentGroup = allPositionGroups[i];

//                        // 遍历当前组的所有位置
//                        for (int j = 0; j < currentGroup.size() && m_running; j++) {
////                            将std::vector<double>转换为vpColVector（假设q是vpColVector）
//                            vpColVector q(6);
//                            for (int k = 0; k < 6; k++) {
//                                q[k] = currentGroup[j][k];
//                            }

//                            m_robot->setPosition(vpRobot::JOINT_STATE, q);
//                            if (j==3) {
//                                QThread::sleep(5);
//                            }

//                            // 可以根据需要添加延时
//                             QThread::sleep(1);

//                            if (!m_running) break;
//                        }
//                    } else {
//                        qDebug() << "索引i超出范围，i =" << i;
//                    }
//                } catch (const vpException &e) {
//                    qDebug() << "控制机械臂时发生错误:" << QString::fromStdString(e.getStringMessage());
//                    m_robot->setRobotState(vpRobot::STATE_STOP);
//                }

//                m_robot->setPosition(vpRobot::JOINT_STATE, m_convergenceJointAngles);
//                QThread::sleep(2);
//            }
//        } catch (const vpException &e) {
//            qDebug() << "控制机械臂时发生错误:" << QString::fromStdString(e.getStringMessage());
//            m_robot->setRobotState(vpRobot::STATE_STOP);
//        }

       // 设置控制模式为位置控制
//       try
//       {
//           m_robot->setRobotState(vpRobot::STATE_POSITION_CONTROL);
//           // 依次移动到四个螺栓位置
//            for (int i = 0; i < targetPoses.size() && m_running; i++){

//                // 移动到当前螺栓位置
//                m_robot->setPosition(vpRobot::END_EFFECTOR_FRAME, targetPoses[i]);
//                QThread::sleep(5);
//                if (!m_running) break;
//                m_robot->setPosition(vpRobot::JOINT_STATE, m_convergenceJointAngles);
//                QThread::sleep(2);
//                if
//                try
//                {   q[0] = vpMath::rad(111.03); q[1] = vpMath::rad(-121.07); q[2] = vpMath::rad(128.24);
//                    q[3] = vpMath::rad(-91.97); q[4] = vpMath::rad(97.61); q[5] = vpMath::rad(167.50);
//                    m_robot->setPosition(vpRobot::JOINT_STATE, q);

//                    q[0] = vpMath::rad(117.03); q[1] = vpMath::rad(-102.59); q[2] = vpMath::rad(137.20);
//                    q[3] = vpMath::rad(-44.15); q[4] = vpMath::rad(100.92); q[5] = vpMath::rad(167.48);
//                    m_robot->setPosition(vpRobot::JOINT_STATE, q);

//                    q[0] = vpMath::rad(124.31); q[1] = vpMath::rad(-78.89); q[2] = vpMath::rad(131.44);
//                    q[3] = vpMath::rad(-53.32); q[4] = vpMath::rad(110.34); q[5] = vpMath::rad(184.02);
//                    m_robot->setPosition(vpRobot::JOINT_STATE, q);


//                    q[0] = vpMath::rad(124.17); q[1] = vpMath::rad(-76.66); q[2] = vpMath::rad(130.52);
//                    q[3] = vpMath::rad(-51.74); q[4] = vpMath::rad(109.35); q[5] = vpMath::rad(184.02);
//                    m_robot->setPosition(vpRobot::JOINT_STATE, q);


//                    q[0] = vpMath::rad(124.31); q[1] = vpMath::rad(-78.89); q[2] = vpMath::rad(131.44);
//                    q[3] = vpMath::rad(-53.32); q[4] = vpMath::rad(110.34); q[5] = vpMath::rad(184.02);
//                    m_robot->setPosition(vpRobot::JOINT_STATE, q);

//                    q[0] = vpMath::rad(117.03); q[1] = vpMath::rad(-102.59); q[2] = vpMath::rad(137.20);
//                    q[3] = vpMath::rad(-44.15); q[4] = vpMath::rad(100.92); q[5] = vpMath::rad(167.48);
//                    m_robot->setPosition(vpRobot::JOINT_STATE, q);

//                    q[0] = vpMath::rad(111.03); q[1] = vpMath::rad(-121.07); q[2] = vpMath::rad(128.24);
//                    q[3] = vpMath::rad(-91.97); q[4] = vpMath::rad(97.61); q[5] = vpMath::rad(167.50);
//                    m_robot->setPosition(vpRobot::JOINT_STATE, q);

//                    q[0] = vpMath::rad(152.23); q[1] = vpMath::rad(-84.77); q[2] = vpMath::rad(87.13);
//                    q[3] = vpMath::rad(-114.92); q[4] = vpMath::rad(96.66); q[5] = vpMath::rad(121.89);
//                    m_robot->setPosition(vpRobot::JOINT_STATE, q);
//                } catch (const vpException &e){
//                    qDebug() << "控制机械臂时发生错误:" << QString::fromStdString(e.getStringMessage());
//                    m_robot->setRobotState(vpRobot::STATE_STOP);
//                }

//            }
             //m_robot->setPosition(vpRobot::END_EFFECTOR_FRAME, flange_1target_pose);
//           // qDebug() << "机械臂已成功移动到目标位姿";
//       }catch (const vpException &e){
//           qDebug() << "控制机械臂时发生错误:" << QString::fromStdString(e.getStringMessage());
//           m_robot->setRobotState(vpRobot::STATE_STOP);
//       }

        // ******************************************伺服结束后的收尾工作***************************************************
        std::cout << "Stop the robot " << std::endl;
        m_robot->setRobotState(vpRobot::STATE_STOP);

        if (opt_plot && plotter != nullptr) {
            delete plotter;
            plotter = nullptr;
        }

        // 在伺服结束时关闭文件
        if (data_file.is_open()) {
            data_file.close();
            qDebug() << "数据记录已完成";
        }

        if (!final_quit) {
            while (!final_quit) {
                m_rs->acquire(color);
                vpDisplay::display(color);

                vpDisplay::displayText(color, 20, 20, "Click to quit the program.", vpColor::red);
                vpDisplay::displayText(color, 40, 20, "Visual servo converged.", vpColor::red);

                if (vpDisplay::getClick(color, false)) {
                    final_quit = true;
                }

                vpDisplay::flush(color);
            }
        }
        if (traj_corners) {
            delete[] traj_corners;
        }

        emit servoStatusChanged("Servo Stopped");

        m_robot->setRobotState(vpRobot::STATE_STOP);
    } catch (const vpException &e) {
        m_robot->setRobotState(vpRobot::STATE_STOP);
        emit errorOccurred(QString("ViSP error: %1").arg(e.what()));

        std::cout << "ViSP exception: " << e.what() << std::endl;
        std::cout << "Stop the robot " << std::endl;
    } catch (const std::exception &e) {
        emit errorOccurred(QString("Error: %1").arg(e.what()));

        std::cout << "ur_rtde exception: " << e.what() << std::endl;
    }

    m_running = false;
}


