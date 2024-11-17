#ifndef DESIGN_OUTPOST_H
#define DESIGN_OUTPOST_H
#include "../armor_detector/armor.h"
#include "../armor_detector/ArmorDetetion.h"
#include "../debug.h"
#include "opencv2/opencv.hpp"
#include<iostream>
#include<vector>

using namespace std;
using namespace cv;

class AngleSolver
{
public:
    AngleSolver();

    /*
     * SetWorldPoints 设置世界坐标点
    */

    void SetworldPoints();

};

#endif //DESIGN_OUTPOST_H
