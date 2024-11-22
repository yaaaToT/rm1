#include "../../include/armor_detector/armor.h"
#include "../../include/outpost/outpost.h"
#include<iostream>
#include "opencv2/opencv.hpp"


cv::Point2f PoseSolver::antiTop(std::vector<ArmorBlob> &armors, double delta_t,/* const SerialPortData &imu_data,
                                                            SerialPort *SerialPort_, */bool &getCenter){
    //DLOG(WARNING) << "  >>>>>>>>>>>>>>>            IN ANTITOP            <<<<<<<<<<<<<<<<" << endl;
    static int frameCount=0;
    static int lostInShootZone=0;
    getCenter=false;
    //for(Armor &a:armors)
        //solveArmor(a,/*imu_data*/);


    if(right_clicked == 0) {
                getCenter = false;
                last_right_clicked = right_clicked;
                return cv::Point2f (0,0);
            }
    if (last_right_clicked == 0 && right_clicked == 1)
                first = true;
    last_right_clicked = right_clicked;
    if (first) { // 只要第一次按右键就清空
                first = false;
                //armors_set.clear();
                //DLOG(WARNING) << ">>>>>>>>>>>init ANTITOP<<<<<<<<<<<<<" << endl;
                timeInZone.clear();
                time.clear();
                getCenter = false;
                return cv::Point2f(0,0);
            }
}




