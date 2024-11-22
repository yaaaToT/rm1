//
// Created by johan on 2024/4/29.
//
#include"opencv4/opencv2/opencv.hpp"
#include"../armor_detector/armor.h"

void FillHole(cv::Mat srcBw, cv::Mat& dstBw);

cv::Mat Preprocess(cv::Mat & src,int enemy_color)
{
    cv::Mat dst;
    cv::Mat src_show=src.clone();
    //Mat element = getStructuringElement(MORPH_ELLIPSE,cv::Size(5,5));
    cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5,5));
    if(enemy_color==RED)
    {
        cv::Mat gray_threshold;
        std::vector<cv::Mat> splited_channels;
        split(src_show,splited_channels);
        cvtColor(src_show,gray_threshold,cv::COLOR_BGR2GRAY);
        threshold(gray_threshold,gray_threshold,96,255,cv::THRESH_BINARY);

        cv::Mat subtract_threshold;
        subtract(splited_channels[2],splited_channels[0],subtract_threshold);
        threshold(subtract_threshold,subtract_threshold,112,255,cv::THRESH_BINARY);

        dilate(subtract_threshold,subtract_threshold,element);
        dst = subtract_threshold & gray_threshold;

        FillHole(dst,dst);

//        std::cout<<"red mode"<<std::endl;
    }
    else if (enemy_color==BLUE)
    {
        cv::Mat gray_threshold;
        std::vector<cv::Mat> splited_channels;
        split(src_show,splited_channels);
        cvtColor(src_show,gray_threshold,cv::COLOR_BGR2GRAY);
        threshold(gray_threshold,gray_threshold,170,255,cv::THRESH_BINARY);

        cv::Mat subtract_threshold;
        subtract(splited_channels[0],splited_channels[2],subtract_threshold);
        threshold(subtract_threshold,subtract_threshold,110,255,cv::THRESH_BINARY);

        dilate(subtract_threshold,subtract_threshold,element);
        dst = subtract_threshold & gray_threshold;

        FillHole(dst,dst);
//        std::cout<<"blue mode"<<std::endl;
    }
    else if(enemy_color==PURPLE)
    {
//        std::cout<<"purple mode"<<std::endl;
    }
    return dst;
}

void FillHole(cv::Mat srcBw, cv::Mat& dstBw)
{
    cv::Mat broedMat = cv::Mat::zeros(srcBw.rows + 2, srcBw.cols + 2, CV_8UC1);
    srcBw.copyTo(broedMat(cv::Range(1, srcBw.rows + 1), cv::Range(1, srcBw.cols + 1)));
    cv::floodFill(broedMat, cv::Point(0, 0), cv::Scalar(255));
    cv::Mat cutMat;
    broedMat(cv::Range(1, srcBw.rows + 1), cv::Range(1, srcBw.cols + 1)).copyTo(cutMat);
    dstBw = srcBw | (~cutMat);
}
