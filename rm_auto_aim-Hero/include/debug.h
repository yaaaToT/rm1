//
// Created by johan on 2024/4/29.
//

#ifndef DESIGN_DEBUG_H
#define DESIGN_DEBUG_H

//#define camera_yaml_path "/home/nuc/rm_2024_thread_energy/params/CameraParams_MindVision.yml"
#define camera_yaml_path "/home/yaaa/rm_auto_aim-Hero/params/CameraParams_MindVision.yml"
//#define onnx_path "/home/nuc/rm_2024_thread_energy/params/fc.onnx"
#define onnx_path "/home/yaaa/rm_auto_aim-Hero/params/fc.onnx"
//#define label_path "/home/nuc/rm_2024_thread_energy/params/label.txt"
#define label_path "/home/yaaa/rm_auto_aim-Hero/params/label.txt"
#define image_width 1280
#define image_height 1024

enum IFKF{
    CLOSE_KF,       // 不使用卡尔曼
    OPEN_KF,        // 使用卡尔曼
};

enum Mode{
    NONE,           // 不用
    ARMOR_MODE,     // 装甲板模式
    ENERGY_MODE,    // 能量机关模式
    BASE_MODE,      // 基地模式
    SHOT_MODE,      // 吊射模式

};

enum ArmorType{
    BIG,
    SMALL
};

enum EnemyColor{
    RED,
    BLUE,
    PURPLE
};


#endif //DESIGN_DEBUG_H

