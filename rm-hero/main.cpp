#include "include/armor_detector/ArmorDetetion.h"
//#include "include/Energy/Energy.h"
#include "include/kalmanfilter/KalmanFilter.h"
#include "include/Preprocess/Preprocess.h"
#include "include/Resolver/Resolver.h"
//#include "include/SerialPort/SerialPort.h"
#include "include/Tracker/Tracker.h"
#include "include/outpost/outpost.h"
//#include "include/mindvision/camera_mindvision_running.h"
//#include "opencv4/opencv2/opencv.hpp"
#include "opencv2/opencv.hpp"
#include <iostream>
#include <ctime>


pthread_t ImagePro;
//pthread_t Serial;

//Mat              src;
cv::Mat              Preprocess_image;
Trackers         track_obj;     // 追踪器对象
Resolver         resolver_obj;  // 解算器对象
//Kalmanfilter     kalman;        // 卡尔曼对象
//MindvisionCamera camera;        // 相机对象
ArmorDetetion    bridge_link;   // 自瞄处理器对象

// 串口对象
//Uart InfoPort;
int  fd_serial    = 0;
bool serial_state = 0;

bool ishave_track = false;

//HostComputerData  RobotInfo;        // 发送
//GroundChassisData MainControlInfo;  // 接收

double Yaw=0.0;
double Pitch=0.0;
double Distance=0.0;

float time_ = 0;

int init_sign = 0;

void* ImageProcess(void*)
{
    //camera.runCamera();
    //测试
    cv::VideoCapture cap("/home/yaaa/下载/mmexport1717936871988(1).mp4");
    
    //VideoCapture cap("/home/yaaa/下载/2b9f0543-a397-4cfb-a031-cd3c563483c8.mp4");
    //VideoCapture cap("/home/yaaa/下载/f37bdc35-7d1f-47e7-aae0-d45d249f72f4.mp4");

    //前哨站
    //VideoCapture cap("/home/yaaa/下载/0.mp4");


    time_t now_time;
    struct tm * timeinfo;
    char buffer[80];
    time(&now_time);
    timeinfo = localtime(&now_time);
    strftime(buffer, sizeof(buffer),"%Y%m%d_%H%M%S",timeinfo);
    //string output_path = "/home/nuc/rm_2024_thread_energy/video/video_" + string(buffer) + ".avi";
    //string output_path = "/home/yaaa/下载/mmexport1717936871988.mp4";
    //writer.open(output_path,VideoWriter::fourcc('M','J','P','G'),60,Size(1280,1024),true);

    int frameWidth=1280/*cap.get(CAP_PROP_FRAME_WIDTH)*/;
    int frameHeight=1024/*cap.get(CAP_PROP_FRAME_HEIGHT)*/;

    //VideoWriter writer(output_path,VideoWriter::fourcc('a','v','c','1'),60,Size(frameWidth,frameHeight),true);
    cv::Mat src;
    while (true)
    {
        cv::Mat                                   temp(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));
        std::chrono::steady_clock::time_point time1 = std::chrono::steady_clock::now();
       
        //camera.Do();
        //camera.getImage(src);
        
        cap>>src;


        //cap>>src;

        //writer.write(src);

        if (!src.empty())
        {

            bridge_link.Image_init(src);

            // 串口读取数据
//            InfoPort.GetMode(fd_serial, MainControlInfo);
            /****
             *
             * 模式选择判断
             *
             */

            // 自瞄模式
//            MainControlInfo.mode = ENERGY_MODE;
//            MainControlInfo.mode = ARMOR_MODE;
            //if (MainControlInfo.mode == ARMOR_MODE)
            //{

                // 图像预处理
                Preprocess_image= Preprocess(src,RED);      // 强制红色模式
                //Preprocess_image = Preprocess(src, BLUE);  // 强制蓝色模式
                // Preprocess_image= Preprocess(src,static_cast<int>(MainControlInfo.color) );     // 串口接收模式
                // 寻找灯条
                bridge_link.FindLight(Preprocess_image);
                // 寻找装甲板
                bridge_link.FindArmor(Preprocess_image);
                // 识别数字
                bridge_link.Digital_recognition(src, bridge_link.armors_, bridge_link.target_armors);
                std::cout << "target: " << bridge_link.target_armors.size() << std::endl;
                if (bridge_link.target_armors.size() != 0)
                {
                    // 确定目标装甲板
                    /***
                     * 目标装甲板数量判断，若只有一个则直接选择击打该装甲板
                     * 若有多个装甲板，
                     * 1.评分模式
                     * 2.距离最近模式
                     */
                    //RobotInfo.if_shoot = 1;

                    bridge_link.Target_Confirm(src, bridge_link.target_armors);
                    // 追踪器判断

                    imshow("src", src);
                    track_obj.TrackJudge(src);

                    //测量每个目标装甲版的距离和角度
                    for (auto& target_armor : bridge_link.target_armors)
                    {
                        resolver_obj.DistanceMeasurer(target_armor);
                    }

                    // 解算弹道
                    resolver_obj.AimTraget(bridge_link.target_armors);

                    /*RobotInfo.Yaw.f   = resolver_obj.send_yaw * 1.0f;
                    RobotInfo.Pitch.f = resolver_obj.send_pitch * 1.0f;
                    RobotInfo.distance.f = resolver_obj.send_distance;*/
                    //double distance=0.0;
                    Yaw = resolver_obj.send_yaw * 1.0f;
                    Pitch= resolver_obj.send_pitch * 1.0f;
                    Distance= resolver_obj.send_distance;

                    track_obj.last_send_yaw   = Yaw;
                    track_obj.last_send_pitch = Pitch;
                }
                else  // 掉帧模式
                {
                    track_obj.armor_drop_count--;
                    if (track_obj.armor_drop_count <= 5 && track_obj.armor_drop_count > 0)
                    {
                        /*putText(bridge_link.src_, "Lost Frame Mode", cv::Point2f(500, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                        RobotInfo.Yaw.f   = track_obj.last_send_yaw;
                        RobotInfo.Pitch.f = track_obj.last_send_pitch;*/

                        putText(bridge_link.src_, "Lost Frame Mode", cv::Point2f(500, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                        Yaw  = track_obj.last_send_yaw;
                        Pitch= track_obj.last_send_pitch;

                    }
                    if (track_obj.armor_drop_count < 0)
                    {
                        track_obj.if_first_frame = true;
                        Yaw        = 0.f;
                        Pitch       = 0.f;
                        //RobotInfo.if_shoot       = 0;
                    }
                }

                    /*RobotInfo.Yaw.f= RobotInfo.Yaw.f * 0.8f;
                    RobotInfo.Pitch.f= RobotInfo.Pitch.f;*/

                Yaw= Yaw * 0.8f;
                Pitch=Pitch;



//                std::cout << "send pitch" << RobotInfo.Pitch.f << std::endl;
//                std::cout << "send yaw" << RobotInfo.Yaw.f << std::endl;
//                putText(src, "Yaw :" + std::to_string(RobotInfo.Yaw.f), Point(4, 45), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
//                putText(src, "Pitch :" + std::to_string(RobotInfo.Pitch.f), Point(4, 65), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
//                putText(src, "distance :" + std::to_string(RobotInfo.distance.f), Point(4, 85), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

                std::cout << "send pitch" << Pitch << std::endl;
                std::cout << "send yaw" << Yaw << std::endl;
                putText(src, "Yaw :" + std::to_string(Yaw), cv::Point(4, 45), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                putText(src, "Pitch :" + std::to_string(Pitch), cv::Point(4, 65), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
                putText(src, "distance :" + std::to_string(Distance), cv::Point(4, 85), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);


                /***
                 *
                 * 是否使用卡尔曼
                 *
                 */

                // 卡尔曼
                /*if (OPEN_KF)
                {
                    if (RobotInfo.Yaw.f != 0.0f && RobotInfo.Pitch.f != 0.0f)
                    {
                        resolver_obj.bullet_fight_time = ((resolver_obj.send_distance * 0.001f) / 30.f);
                        kalman.gain_yaw                = MainControlInfo.gain_yaw.f * 1.55;
                        kalman.send_pitch              = MainControlInfo.gain_pitch.f * 1.25;

                        kalman.send_yaw   = RobotInfo.Yaw.f;
                        kalman.send_pitch = RobotInfo.Pitch.f;

                        if (track_obj.if_first_frame == true)
                        {
                            track_obj.if_first_frame = false;
                            kalman.InitKalman();
                        }
                        else
                        {
                            kalman.UpDate(time_);
                        }
                        kalman.KalmanPredict(resolver_obj.bullet_fight_time * 1.2f);

                        RobotInfo.Yaw.f      = kalman.send_yaw;
                        RobotInfo.Pitch.f  = kalman.send_pitch;
                        RobotInfo.distance.f= resolver_obj.send_distance / 1000.0;
                    }
                }
                imshow("klsrc", src);
                imshow("dst", Preprocess_image);*/
            //}
            // 能量机关模式
            /*else if (MainControlInfo.mode == ENERGY_MODE)
            {
                AngleSolver*       angleslover = new AngleSolver();
                AutoAimgingEnergy* energyfind  = new AutoAimgingEnergy(angleslover);

                unsigned char color;
//                if (MainControlInfo.color == 1)
//                {
//                    color = 0;
//                }
//                else if (MainControlInfo.color == 0)
//                {
//                    color = 1;
//                }
//                else
//                {
//                   color = 0;
//                }
//                color = 0;  //强制转换成红色
                energyfind->AimingEnergy(src, color, RobotInfo, MainControlInfo);

		RobotInfo.Yaw.f      = 0;
                RobotInfo.Pitch.f    = 0;
            }
            else if(MainControlInfo.mode == NONE)
            {
                RobotInfo.Yaw.f      = 0;
                RobotInfo.Pitch.f    = 0;
            }
            else{

            }*/

                /*Mat roi = src(cv::Rect(x, y, roiWidth, roiHeight));
                cv::Mat enlargedRoi;
                cv::resize(roi, enlargedRoi, cv::Size(roiWidth * 2, roiHeight * 2));

                enlargedRoi.copyTo(src(cv::Rect(x, y, enlargedRoi.cols, enlargedRoi.rows)));*/

            std::chrono::steady_clock::time_point time2     = std::chrono::steady_clock::now();
            std::chrono::duration<double>         time_used = std::chrono::duration_cast<std::chrono::duration<double>>(time2 - time1);
            time_                                           = static_cast<float>(time_used.count());
            double fps                                      = 1.0 / time_;

        putText(src, "fps :" + std::to_string(fps), cv::Point(4, 105), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);


        circle(src, cv::Point2f(src.cols / 2, src.rows / 2), 10, cv::Scalar(0, 255, 0), 1, 3);
            //line(src,Point(640,0),Point(640,1024),Scalar(0,255,0));
            //line(src,Point(0,512),Point(1280,512),Scalar(0,255,0));

            /*line(src,Point(350,250),Point(350,500),Scalar(0,255,0));
            line(src,Point(160,250),Point(160,500),Scalar(0,255,0));

            line(src,Point(160,250),Point(350,250),Scalar(0,255,0));
            line(src,Point(160,500),Point(350,500),Scalar(0,255,0));*/
            
            //writer.write(src);
            
            imshow("src", src);
            if(cv::waitKey(30)==27) break;
            //InfoPort.TransformTarPos(fd_serial, RobotInfo);
        }
        /*else
        {
            std::cout << "frame empty" << std::endl;
        }*/

    }
    cap.release();
    //destroyAllWindows();
    //writer.release();
    //CameraUnInit(camera.hCamera);
    //free(camera.g_pRgbBuffer);
    cv::destroyAllWindows();
}

/*void* GetData(void*)
{
    serial_state            = InfoPort.Init_serial(fd_serial, 115200) + 1;
    RobotInfo.if_real_shoot = 0;
    int num_=0;

    while (1)
    {
        num_++;
        if (num_ >= 1000000 && init_sign==0)
        {
            num_=0;

            init_sign = 1;
        }

        if (serial_state)
        {
            if(init_sign==1)
            {
                InfoPort.close_serial(fd_serial);
                InfoPort.Init_serial(fd_serial, 115200);
                init_sign=0;
            }

            InfoPort.GetMode(fd_serial, MainControlInfo);

            if (fabs(RobotInfo.Pitch.f) < 1.5)
            {
                if (RobotInfo.if_shoot == 1 && fabs(RobotInfo.Yaw.f) < 4 && RobotInfo.distance.f < 1.7)
                {
                    RobotInfo.if_real_shoot = 1;
                }
                else if (RobotInfo.if_shoot == 1 && fabs(RobotInfo.Yaw.f) < 3.0 && RobotInfo.distance.f < 3.5 && RobotInfo.distance.f >= 1.7)
                {
                    RobotInfo.if_real_shoot = 1;
                }
                // yaw <1.9  2.9  1.7
                else if (RobotInfo.if_shoot == 1 && fabs(RobotInfo.Yaw.f) < 1.9 && RobotInfo.distance.f >= 3.5 && RobotInfo.distance.f < 5)
                {

                    RobotInfo.if_real_shoot = 1;
                }
                else if (RobotInfo.if_shoot == 1 && fabs(RobotInfo.Yaw.f) < 1.8 && RobotInfo.distance.f > 5)
                {
                    RobotInfo.if_real_shoot = 1;
                }
                else
                {
                    RobotInfo.if_real_shoot = 0;
                }
            }

            // 死区限制
            if ( (fabs(RobotInfo.Yaw.f) > 25.0) || (isnan(RobotInfo.Yaw.f)))
            {
                RobotInfo.Yaw.f = 0;
            }
            if ( (fabs(RobotInfo.Pitch.f) > 25.0) || (isnan(RobotInfo.Pitch.f)))
            {
                RobotInfo.Pitch.f = 0;
            }
            //发送数据
            RobotInfo.if_real_shoot = 0;
            RobotInfo.if_shoot = 0;
            InfoPort.TransformTarPos(fd_serial, RobotInfo);
        }
    }
}*/

int main()
{
    pthread_create(&ImagePro, nullptr, ImageProcess, nullptr);
    //pthread_create(&Serial, nullptr, GetData, nullptr);

    pthread_join(ImagePro, nullptr);
    //pthread_join(Serial, nullptr);


    return 0;
}

