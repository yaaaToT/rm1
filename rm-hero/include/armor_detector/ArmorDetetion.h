
#ifndef DESIGN_ARMORDETETION_H
#define DESIGN_ARMORDETETION_H
#include "armor.h"
#include "../outpost/params.h"
#include "../outpost/outpost.h"
#include "../Tracker/Predictor.h"

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

class ArmorDetetion {
public:
    ArmorDetetion();
    void Image_init(cv::Mat& src){src_=src.clone();};
    // 寻找灯条
    bool FindLight(cv::Mat &src);
    bool IsLight(const Light& light);
    // 寻找装甲板
    bool FindArmor(cv::Mat &src);
    bool IsArmor(Armor & armor);

    // 数字识别
    void Digital_recognition(const cv::Mat& src, std::vector<Armor> armors, std::vector<Armor>& target_armors);
    void extractNumbers(const cv::Mat& src, std::vector<Armor>& armors);
    void classify(std::vector<Armor>& armors, std::vector<Armor>& target_armors);

    // 确定目标装甲板
    bool Target_Confirm(cv::Mat& src, std::vector<Armor>& target_armors);
    // 识别到的灯条
    std::vector<Light> lights_;
    // 识别到的所有装甲板（未经过数字识别判断）
    std::vector<Armor> armors_;
    // 确定是的装甲板
    std::vector<Armor> target_armors;
    // 目标装甲板
    Armor confirm_armor;

    // 图像识别属性
    EnemyColor enemy_color;

    // 原始图像
    cv::Mat src_;

private:
    // 数字识别属性
    cv::dnn::Net net_;
    std::vector<std::string> class_names_;
    std::vector<std::string> ignore_classes_;

};



//*********************************************************************************



//static Ptr<cv::Tracker>tracker;

struct Params_ToDecter{
//    Image**frame_pp;
    GroundChassisData*SerialPortData_; // 传出的串口数据
    bool is_update;

    Params_ToDecter(){
//        frame_pp=nullptr;

        SerialPortData_=new GroundChassisData();
        is_update=false;
    }
};

class Detector{
public:
    explicit Detector();
    ~Detector()=default;
//    void setParams(const Params_ToVideo &params_to_video,const Params_ToSerialPort &params_to_serial_port);
    void startDetect(const Params_ToDecter &param,GroundChassisData*SerialPort_);
    void outpostMode();
private:
    inline void calcGammaTable(float gamma);
//    inline void drawArmorCorners(Mat&drawing,const ArmorBlob&armor,const Scalar&color);
    double fitTrajectory(const cv::Point3d&trans,double v);
//    NumberClassifier classifier=NumberClassifier(0);
//    unordered_map<int,float>gamma_table;
    Params_ToDecter _detector_thread_params;
//    ROIAccelerator *roi_accelerator;
//            TargetChooser *target_chooser;

            // TODO 增加一个udp发送，用于调试
            UDPSender * udpsender;
            PoseDataFrame poseDate;
            DeltatFrame delta_t_frame;
};


#endif //DESIGN_ARMORDETETION_H
