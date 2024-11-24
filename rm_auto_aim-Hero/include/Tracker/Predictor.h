#ifndef PREDICTOR_H
#define PREDICTOR_H

#include "../SerialPort/SerialPort.h"
#include "../SerialPort/UDPSender.h"
#include "eigen3/Eigen/Dense"
#include<vector>
#include "opencv2/opencv.hpp"
#include "opencv2/core/eigen.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

typedef struct
    {
        float pitch;    // rad
        float yaw;      // rad
        float time;     // 击打弹道时间(ms)
        float distance; // 距离
    } Angle_t;

    class Predictor
    {
    public:
        Predictor();
        ~Predictor();

        Angle_t Predict(const cv::Point3d &armor1, const cv::Point3d &armor2, bool is_get_second_armor, int detect_mode, /*SerialPortData SerialPortData_,*/ float &frame_delta_t);
        void resetPredictor();

        // vector<Eigen::Vector3d> getArmorSerial() { return vehicle_tracker->getArmorSerial(); };
        // Eigen::Vector3d getPredictPoint() { return predict_hit_point; };
        // vector<float> getSpeed() { return vehicle_tracker->speed_vector; };
        // float getYaw() { return vehicle_tracker->getYaw(); };
        // bool getArmorSwitch() { return vehicle_tracker->armor_switch; }

        // float calShootTime(const Eigen::Vector3d &armor_trans, SerialPortData SerialPortData_);
        AimFrame getDebugData() { return data; }
        // VehicleFrame getDebugData2() { return data2; }
        UDPSender* udpsender;
    private:

        Angle_t ballistic_equation(float gim_pitch, cv::Point3d armor_Position);
        float ShootSpeed;

        Angle_t shootAngleTime_now;
        Angle_t shootAngleTime_pre;
        // VehicleTracking *vehicle_tracker;

        // 预测击打点
        // Eigen::Vector3d predict_hit_point;

        // 扩展卡尔曼滤波
//        NormalEKF *ekf_filter;

        cv::Point3d last_pose_vec;

        AimFrame data;
        // VehicleFrame data2;
    };

#endif // PREDICTOR_H
