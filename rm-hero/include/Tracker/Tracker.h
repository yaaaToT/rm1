
#ifndef DESIGN_TRACKER_H
#define DESIGN_TRACKER_H
#include "../outpost/params.h"
#include "../kalmanfilter/KalmanFilter.h"
#include "eigen3/Eigen/Dense"
#include "Eigen/SVD"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"
#include <vector>
#include "opencv2/opencv.hpp"

class Trackers {
public:
    void TrackJudge(cv::Mat& src);

    int armor_lost_count;       //装甲板丢失帧数
    int armor_drop_count;       //装甲板掉帧帧数
    int armor_found_count;      //装甲板发现帧数

    float last_send_yaw;
    float last_send_pitch;
    float last_send_distance;

    int tracker_count;          //跟踪计数


    bool if_track;              // 是否跟踪标志
    bool if_armors_found;       // 是否发现装甲板
    bool if_first_frame;        // 是否为第一帧


};

class VehicleTracking
{
public:
    VehicleTracking();
    ~VehicleTracking();
    Eigen::Vector3d predictVehicleState(const Sophus::SE3d &armor_pose,const Sophus::SE3d &armor_pose_sec,bool is_get_second_armor,int &detect_mode,float shoot_time,float frame_delta_t,float yaw_gimbal);
    void resetTracker();
    std::vector<Eigen::Vector3d> getArmorSerial(){return armor_serial;}
    float getYaw(){return yaw_send;};
    Eigen::Matrix<double,9,1>getVehicleState(){return vehicle_state;}
    std::vector<float>speed_vector=std::vector<float>(3);
    bool armor_switch=false;

private:
    ExtendedKalman<double,9,4>*extended_kalman_filter; // 状态向量9维，x_c,v_x,y_c,v_y,z_c,v_z,yaw,v_yaw,r ,测量四维x,y,z,yaw
    void setUpdateTime(const double &delta_t);

    Eigen::Vector3d getPredictPoint(const Eigen::Matrix<double,9,1>whole_car_state,float &shoot_time,float yaw_gimbal);

    Eigen::Vector4d PoseToMeasurement(const Sophus::SE3d &pose);

    // 设置观测向量,x_1,y_1,z_1,yaw_1
    void setMeasurementVector();
    void setQandRMatrix();
    void getVehicleState(Eigen::Vector4d &measure);
    Eigen::Vector3d getArmorPositionFromState(Eigen::Matrix<double,9,1>);
    void handleArmorJump(Eigen::Vector4d &measure);
    void rebootKalman(bool,Eigen::Vector4d,Eigen::Vector4d);

    void setTransitionMatrix();
    void setObservationMatrix();

    double update_time;
    bool is_kalman_init;

    // 状态相量9维,x_c,v_x,y_c,v_y,z_c,v_z,yaw,v_yaw,r
    Eigen::Matrix<double,9,1>vehicle_state;
    std::vector<Eigen::Vector3d>armor_serial;
    float yaw_send;
    double last_z,last_r,last_yaw;
};


#endif //DESIGN_TRACKER_H
