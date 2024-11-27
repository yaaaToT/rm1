#ifndef DESIGN_OUTPOST_H
#define DESIGN_OUTPOST_H

#include "../armor_detector/armor.h"
#include "../armor_detector/ArmorDetetion.h"
#include "../Tracker/Predictor.h"
#include "../SerialPort/UDPSender.h"
#include "../SerialPort/SerialPort.h"
#include "params.h"
#include "newArray.h"
#include "Array.h"
#include "../debug.h"

#include "Eigen/Dense"
#include "glog/logging.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <string>
#include <chrono>
#include <set>
#include <iostream>
#include <vector>


class PoseSolver
{
public:
    explicit PoseSolver();
    //更新辅瞄
    cv::Point2f antiTop(std::vector<ArmorBlob>&armors,double delta_t,const GroundChassisData&imu_data,Uart*SerialPort_,bool&getTarget);
//    bool getPoseInCamera(std::vector<ArmorBlob> &armors, double delta_t, const SerialPortData imu_data, SerialPort* SerialPort_, int &this_frame_class, int &last_frame_class);
//    cv::Point2f outpostMode(std::vector<ArmorBlob> &armors, double delta_t, const SerialPortData& imu_data, SerialPort* SerialPort_, bool& getTarget);
//    cv::Point2f greenShoot(std::vector<cv::Rect2f> green, double delta_t, const SerialPortData &imu_data, SerialPort *SerialPort_,bool &getCenter);
            bool adjustToPoint(cv::Point2f greenTarget, cv::Point2f cameraTarget, const GroundChassisData &imu_data,
                                           int& modeNum,bool quickStart=false,bool restart=false);
//            cv::Point2f halfoutpostMode(std::vector<ArmorBlob> &armors, double delta_t ,const SerialPortData& imu_data, SerialPort* SerialPort_);
            void sentinelMode(std::vector<ArmorBlob> &armors, double delta_t ,const GroundChassisData& imu_data, Uart* SerialPort_);
            void clearCircle();
            void clearSentinel();
            cv::Point2f reproject(cv::Point3f center);  // 将中心进行重投影
            void update_delta_t(double &delta_time);

private:
            void setCameraMatrix(double fx, double fy, double u0, double v0);
                    void setDistortionCoefficients(double k_1, double k_2, double p_1, double p_2, double k_3);
                    void setimu(float pitch, float yaw, float roll);

                    Sophus::SE3d solveArmor(ArmorBlob& armors,const GroundChassisData& imu_data);
                    void solveArmorBA(ArmorBlob& armors,const GroundChassisData& imu_data);
                    float calcDiff(const ArmorBlob& a, const ArmorBlob& b);
                    float cosineLaw(float a, float b, float c);
                    int chooseArmor(const std::vector<ArmorBlob>& armors);


                    float length_of_small = 0.0675f;        // 2倍左右
                    float height_of_small = 0.0275f;
                    float length_of_big = 0.1125f;          // 4倍左右
                    float height_of_big = 0.0275f;
                    // 保存数据到指定txt，用一个file
                    //ofstream file_1, file_2;

                    // 小装甲板3d坐标
                    std::vector<cv::Point3f> points_small_3d = {cv::Point3f(-length_of_small, -height_of_small, 0.f),
                                                       cv::Point3f(length_of_small, -height_of_small, 0.f),
                                                       cv::Point3f(length_of_small, height_of_small, 0.f),
                                                       cv::Point3f(-length_of_small, height_of_small, 0.f)};

            //          大装甲板3d坐标
                    std::vector<cv::Point3f> points_large_3d = {cv::Point3f(-length_of_big, -height_of_big, 0.f),
                                                       cv::Point3f(length_of_big, -height_of_big, 0.f),
                                                       cv::Point3f(length_of_big, height_of_big, 0.f),
                                                       cv::Point3f(-length_of_big, height_of_big, 0.f)};

                    Predictor* predictor;

                    // 解算pnp需要的装甲板坐标
                    std::vector<cv::Point3f> points_3d;
                    cv::Vec2d reproject_error;
                    std::vector<float> reprojectionError;
                    std::vector<cv::Mat> tvecs;
                    std::vector<cv::Mat> rvecs;

                    // TODO 增加一个udp发送，用于调试
                            UDPSender *udpsender = new UDPSender("192.168.1.11", 3000);
                            PoseDataFrame outpostPoseDataFrame;
                            CenterFrame centerFrame;

                            cv::Mat camera_matrix;
                            cv::Mat distortion_coefficients;
                            cv::Mat tvec;
                            cv::Mat rvec;
                            cv::Mat m_T;
                            cv::Mat m_R;

                            Sophus::Matrix3d e_R;
                            Sophus::Vector3d e_T;

//                            // armor的yaw方向角度
                            double yaw;

                            // vector<Point3f> points_large_3d;
                            // vector<Point3f> points_small_3d;

                            Sophus::SE3d armor_to_camera;
                            Sophus::SE3d camera_to_gimbal; // imu's world
                            Sophus::SE3d armor_to_gimbal;
                            Sophus::SE3d armor_to_world;
                            Sophus::SE3d gimbal_to_world;

                            Sophus::SE3d camera_to_world; // imu's world

                            // Predictor* predictor;

                            ArmorBlob last_armor;
                            ArmorBlob armor;

   //  ***************************************************************************

//public:
//    std::vector<cv::Point3f> points_in_world;
//    std::vector<cv::Point2f> points_in_camera;

//    cv::Point3f posture; //位姿


    bool right_clicked = true;
    bool last_right_clicked = true;
    bool first = true;
    bool has_same_class = false;
    bool find_outpost = false;
    bool shoot = false;


    int top_pri = 3;

            int top_cnt = 0;
            int lost_cnt = 0;
            std::chrono::time_point<std::chrono::steady_clock> top_begin;
            std::chrono::time_point<std::chrono::steady_clock> top_exit;
            double exit_duration = 0;
            std::chrono::time_point<std::chrono::steady_clock> shoot_begin;
            double shoot_duration = 0;
            std::chrono::time_point<std::chrono::steady_clock> w_begin;
            double w_duration = 0;
            bool w_init = false;

            // double x0, y0, z0, r, angle0;
            cv::Point3d center;
            cv::Point3d cur;
            // double armor_y = 0;

            // 记录
            std::chrono::steady_clock::time_point send_shoot_time;

            // 5ms 每帧，0.2r/s
             RollingArray<cv::Point3d> circle = RollingArray<cv::Point3d>(100);

            // 获得5次
            RollingArray<cv::Point3d> outpost_center = RollingArray<cv::Point3d>(40);   // 取10次的中心的

            RollingArray<int> filtered_pitch = RollingArray<int>(100);

            // 创建outpost
            newRollingArray outpost = newRollingArray(120);     // 100fps

            RollingArray<double> roll = RollingArray<double>(100);
            newIntRollingArray thresh = newIntRollingArray(20);  // 为了找到比较低的哪一个

            RollingArray<cv::Point3d> sentinel = RollingArray<cv::Point3d>(100);

            float delta_t;
            ceres::CostFunction *cost = nullptr;
            ceres::LossFunction* loss = nullptr;
            // antiTop
            // 用一个set存储并排序所有的armor
            std::vector<ArmorBlob> armors_set;
            std::vector<std::chrono::steady_clock::time_point> time;
            // 用一个vector存储每一个armor到达最近位置的时间
            std::vector<std::chrono::milliseconds> timeInZone;
            // 发弹延迟
            double tmp_time=163; // 由链路决定的发弹延迟

            double time_bias=OutpostParam::time_bias;

            double time_bias_inverse = OutpostParam::time_bias_inverse;

            const std::string target_mode_str[4] = {"NOT_GET_TARGET", "CONTINOUS_GET_TARGET", "LOST_BUMP", "DETECT_BUMP"};
            // TODO 在这里写死了弹丸速度
            double speed = GlobalParam::SHOOT_SPEED;
//            GimbalPose solveGimbalPose(cv::Point3d shootTarget);


};

#endif //DESIGN_OUTPOST_H
