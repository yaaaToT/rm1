#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "opencv2/opencv.hpp"
#include <vector>

class Classifier {
    public:
        virtual int predict(const cv::Mat& frame) = 0;
        virtual void train() = 0;
    };

    class NSVM : public Classifier {
    public:
        NSVM();
        int predict(const cv::Mat& frame) override;
        void train() override;
        const cv::Mat& getArmor();

    private:
        cv::Ptr<cv::ml::SVM> svm;
        cv::Ptr<cv::CLAHE> clahe;

        cv::Mat armor;
        cv::Mat final;
        cv::Mat cls;
        int thresh; // unused
        uchar gamma_table[256];
    };

    class FSVM : public Classifier {
    public:
        FSVM();
        int predict(const cv::Mat& frame) override;
        void train() override;
        const cv::Mat& getArmor();

    private:
        cv::Ptr<cv::ml::SVM> svm;
        cv::HOGDescriptor hog;

        cv::Mat armor;
        std::vector<float> decs;
    };

    class CNN : public Classifier {
    public:
        CNN();
        int predict(const cv::Mat& frame) override;
        int predict(const cv::Mat& frame, float &prob, float prob_thresh);
        void train() override;

    private:
        void softmax(cv::Mat& mat);
        // string model = "../src/utils/tools/armor.bak.onnx";
        std::string model = "/home/yaaa/rm1/rm-hero/params/best_06_02.onnx";
        cv::dnn::Net net;
    };

#endif // CLASSIFIER_H
