#include "../../include/armor_detector/armor.h"
#include "../../include/outpost/outpost.h"
#include "../../include/Tracker/Predictor.h"
#include<iostream>
#include<thread>
#include "opencv2/opencv.hpp"


cv::Point2f PoseSolver::antiTop(std::vector<ArmorBlob> &armors, double delta_t, const SerialPortData &imu_data,
                                                            SerialPort *SerialPort_, bool &getCenter){
    DLOG(WARNING) << "  >>>>>>>>>>>>>>>            IN ANTITOP            <<<<<<<<<<<<<<<<" << std::endl;
    static int frameCount=0;
    static int lostInShootZone=0;
    getCenter=false;
    for(ArmorBlob &a:armors)
        solveArmor(a,imu_data);

    ArmorBlobs candidates;
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
                armors_set.clear();
                DLOG(WARNING) << ">>>>>>>>>>>init ANTITOP<<<<<<<<<<<<<" << std::endl;
                timeInZone.clear();
                time.clear();
                getCenter = false;
                return cv::Point2f(0,0);
            }
    if (last_right_clicked == 0 && right_clicked == 1)
                first = true;
            last_right_clicked = right_clicked;
            if (first) { // 只要第一次按右键就清空
                first = false;
                armors_set.clear();
                DLOG(WARNING) << ">>>>>>>>>>>init ANTITOP<<<<<<<<<<<<<" <<std::endl;
                timeInZone.clear();
                time.clear();
                getCenter = false;
                return cv::Point2f(0,0);
            }
            if (armors.size() < 1) {
                getCenter = false;
                return cv::Point2f(0, 0);
            }

            if (first) { // 按下右键时瞄准中心装甲
                        // 按照距离中心的距离来进行排序，找到最靠近中心的装甲板

                        // 近距离辅瞄
                        sort(armors.begin(), armors.end(), [](const ArmorBlob &a, const ArmorBlob &b) -> bool {
                            const cv::Rect &r1 = a.rect;
                            const cv::Rect &r2 = b.rect;
                            return abs(r1.x + r1.y + r1.height / 2 + r1.width / 2 - 1024 / 2 - 1280 / 2) <
                                   abs(r2.x + r2.height / 2 - 1024 / 2 + r2.y + r2.width / 2 - 1280 / 2);
                        });
                        // 找出距离中心最近的装甲板
                        armor = armors.at(0);
                        // 最中心的装甲板优先级最高, 操作手瞄准的是最高优先级的装甲板 top_pri 就是操作手开启辅瞄时指定的跟踪目标（最靠近中心的部分）
                        top_pri = armor._class;     // 设定优先级
                    }
                    // 中心最近的装甲板
                    // DLOG(WARNING) << "class: " << top_pri << std::endl;
                    // 选择装甲板(优先选‘操作手右击离中心最近的’，其次‘选上一次选中的装甲板’，然后‘选择英雄’，最后选择‘步兵’    优先操作手的选项
                    int target = chooseArmor(armors);
                    // 选中的目标
//                    SerialParam::send_data.num = target; // 当前辅瞄选定的目标
//                    DLOG(WARNING)<<"target: "<<target<<endl;
                    // 筛选出 优先级最高的 类别
                    for (const auto &a: armors) {
                        if (a._class == target)
                            candidates.push_back(a);
                    }

                    // 没有符合的装甲板， 可能是掉帧, 减少错
                    if (candidates.size() < 1) {
            //            getCenter = false;

                        return cv::Point2f(0, 0);
                    }

                    //DLOG(INFO)<<"armors set:" <<armors_set.size()<<endl;

                    if(armors_set.size()>5) {
                        getCenter=true;
                    }
                    else getCenter= false;
                    //DLOG(INFO) << "armor size: " << candidates.size();
                    if (candidates.size() > 1) {
            //            sort(candidates.begin(), candidates.end(), [&](const ArmorBlob &a, const ArmorBlob &b) {
            //                return calcDiff(a, last_armor) < calcDiff(b, last_armor);
            //            });
                        // DLOG(INFO) << "diff: " << calcDiff(candidates.at(0), last_armor) << " " << calcDiff(candidates.at(1), last_armor);
                        // 找出y最小的装甲板
                        sort(candidates.begin(), candidates.end(), [](const ArmorBlob &a, const ArmorBlob &b) {
                            return a.y < b.y;
                        });
                    }

                    // 选择和上一帧识别的装甲板距离最近的那唯一一个装甲板
                    armor = candidates.at(0);

                    //DLOG(WARNING)<<"armor_id: "<< armor._class<<endl;
                    // candidates 都是装甲板和上一帧锁定的装甲板类别一致的
                    // 都更新 last_armor

                    if(target==last_armor._class){
                        // 按照和上一帧出现的装甲板距离进行降序排序（相当于一个追踪的效果）

                        //掉帧缓冲
                        if(lost_cnt<20&&calcDiff(armor,last_armor)>0.3){
                            armor=last_armor;
                            lost_cnt++;
                        }else{
                            lost_cnt=0;
                            armor=candidates.at(0);
                        }
                        bool has_two_armor=false;
                        cv::Point3d armor1=cv::Point3d(candidates[0].x,candidates[0].y,candidates[0].z);

                        cv::Point3d armor2 = cv::Point3d(0, 0, 0);
                        if (candidates.size() >= 2) {
                           has_two_armor = true;
                           armor2 = cv::Point3d(candidates[1].x, candidates[1].y, candidates[1].z);
                         }
                        float delta_time=delta_t;  // 类型转换
                        Angle_t tmp_angle;
                        // 如果candidates[0]的y比shootcenter高超过5cm，不加入
                                    static cv::Point3d shootCenter = cv::Point3d(0, 0, 0);
                        //            if (candidates[0].y - shootCenter.y > 0.08 && norm(shootCenter) != 0 && armors_set.size()>25 && candidates.size() == 1) {
                        //                getCenter = false;
                        //                return Point2f(0, 0);
                        //            }
                         DLOG(WARNING)<<"center y diff: "<<candidates[0].y - shootCenter.y<<std::endl;
                         if(has_two_armor) {
                             DLOG(WARNING)<<"center y diff of armor 2:"<<candidates[1].y-shootCenter.y<<std::endl;
                         }
                         armors_set.push_back(candidates[0]);
                         // 获取现在的时间戳
                         auto now = std::chrono::steady_clock::now();
                         // 获取armors中前20帧近距离的装甲板，取平均值获得中心点，当多于20帧时删除最远的
                          frameCount++;
                           DLOG(INFO)<<"armors_set size: "<<armors_set.size()<<std::endl;
                           if(armors_set.size()>31){
                               //通过迭代器找到最大元素，删除该元素
                               auto max=max_element(armors_set.begin(),armors_set.end());
                               //找到最后一个元素。删除该元素
                               armors_set.erase(armors_set.end());
                               armors_set.erase(max);
                           }
                           DLOG(WARNING) << "              OUT ANTITOP" << std::endl;
                           for(auto a:armors_set){
                               shootCenter+=cv::Point3d(a.x,a.y,a.z);
                           }
                           if(armors_set.size()!=0)
                               shootCenter=shootCenter/(int)armors_set.size();
                           // 显示shootCenter

                           /////////////////////弹道公式
                           // 没有预测，只有跟随

                           shootCenter.y=(shootCenter.y==0?1e-6:shootCenter.y);
                           tmp_angle.yaw=atan(-shootCenter.x/shootCenter.y);
                           tmp_angle.distance=sqrt(shootCenter.x*shootCenter.x+shootCenter.y*shootCenter.y);
                           tmp_angle.pitch=atan(shootCenter.z/tmp_angle.distance);

                           double theta=tmp_angle.pitch;
                           double delta_z;
                           double k1=0.47*1.169*(2*M_PI*0.02125*0.02125)/2/0.041;
                           double center_distance=tmp_angle.distance/0.9;   //距离
                           double gravity=9.8;
                           double flyTime;
                           for(int i=0;i<100;i++){
                               //计算炮弹飞行时间
                               flyTime=(pow(2.718281828,k1*center_distance)-1)/(k1*speed*cos(theta));
                               delta_z=shootCenter.z-speed*sin(theta)*flyTime/cos(theta)+
                                       0.5*gravity*flyTime*flyTime/cos(theta)/cos(theta);
                               if(fabs(delta_z)<0.000001)
                                   break;
                               theta-=delta_z/(-(speed*flyTime)/pow(cos(theta),2)+
                                               gravity*flyTime*flyTime/(speed*speed)*sin(theta)/pow(cos(theta),3));

                           }

                           double tmp_pitch=(theta/M_PI*180)*100;

                           DLOG(WARNING) << "           timeInZone size:" << timeInZone.size() << std::endl;
                           static bool markLast;// 用于记录是否标记了上一个的时间
                           if(armors_set.size()>=30){
                               //使用重投影到平面内平面距离判定击打
                               cv::Point2f shootCenter2D=reproject(shootCenter);
                               cv::Point2f armor2D=reproject(cv::Point3d(candidates[0].x,candidates[0].y,candidates[0].z));

                               double p=0.9;
                               if(abs(imu_data.roll)>500) p=0.7;
                               double distance = p * fabsf(shootCenter2D.x - armor2D.x) + (1 - p) * fabsf(shootCenter2D.y - armor2D.y);
                               DLOG(WARNING)<<"shoot distance: "<<distance<<std::endl;

                               if(lostInShootZone>=3&&!markLast){
                                   std::chrono::steady_clock::duration sum=std::chrono::steady_clock::duration::zero();
                                   for(const auto &timestamp: time){
                                       sum+=timestamp.time_since_epoch();
                                   }
                                   if(time.size()){
                                       std::chrono::steady_clock::duration average=sum/time.size();
                                       auto average_ms=std::chrono::duration_cast<std::chrono::milliseconds>(average);
                                       timeInZone.push_back(average_ms);
                                   }
                                   time.clear();
                                   markLast=true;
                               }
                               // 当装甲板的位置与center距离小于0.05m认为进入了击打范围
                               //                DLOG(WARNING) << "distance: " << distance << endl;
                               if(distance<5.0){
                                   DLOG(WARNING) << "IN SHOOT ZONE!!!!" << std::endl;

                                   markLast = false;
                                   lostInShootZone = 0;
                                   // 将当前时间加入time
                                   time.push_back(now);
                                   // 拟合一圈开始，识别出了四个装甲板，开始计算时间差
                                   if (timeInZone.size() >= 3) {
                                       // 计算timeInZone相邻时间之间差值的平均值
                                       std::chrono::steady_clock::duration sum = std::chrono::steady_clock::duration::zero();


                                        sum = timeInZone[timeInZone.size()-1] - timeInZone[timeInZone.size()-3];
                                        std::chrono::steady_clock::duration average = sum / 2; // 只用最近两个周期的时间进行拟合
                                        auto average_ms = std::chrono::duration_cast<std::chrono::milliseconds>(average);
                                        // 设置延迟击打
                                        SerialParam::send_data.shootStatus = 0;
                                        // 休眠时间，下两个装甲板转过来的时间提前一个发弹延迟的时间
                                        bool shootable = 1;
                                        if(average_ms.count()>950) shootable=0; // 可能存在掉帧导致记录了两个装甲板的时间
                                        DLOG(WARNING) << "average time: " << average_ms.count() << "; tmp_time: " << tmp_time
                                                                         << "; flyTime: " << flyTime * 1000 << "; user bias: "<< (int)SerialParam::recv_data.user_time_bias * 25<< std::endl;
                                        int i=0, sleep_time=-1;

                                        double timeBias;
                                        if(SerialParam::recv_data.outpost_state==0) timeBias=time_bias;
                                        else timeBias=time_bias_inverse;



                                        while (sleep_time <0) {
                                               sleep_time = (int) (1.0 * i * average_ms.count()  - 1. * tmp_time - flyTime * 1000 + timeBias + (int)SerialParam::recv_data.user_time_bias * 25);
                                               i++;
                                         }
                                        // 设置打弹
                                        if(shootable)
                                          std::thread([this, sleep_time, SerialPort_]() {
                                            // 延迟击打
                                          std::this_thread::sleep_for(std::chrono::milliseconds(sleep_time));
                                          SerialParam::send_data.shootStatus = 1;
                                          SerialPort_->writeData(&SerialParam::send_data);
                                          DLOG(INFO) << "                shoot";
                                            }).detach();

                                    }
                               }
                               // 距离大于0.2认为离开了击打范围
                               else if(distance>25.0){
                                   lostInShootZone++;
                               }
                           }
                           // DLOG(INFO) << armor1.z;
                                       DLOG(INFO) << "distance : " << tmp_angle.distance;
                                       // 跟随功能

                                       SerialParam::send_data.shootStatus = 0; // 实际击打

                                       SerialParam::send_data.pitch = tmp_pitch;

                                       double tmp_yaw = tmp_angle.yaw / M_PI * 180 * 100;
                                       tmp_yaw += 0.8 * 100;       // TODO: 进行补偿
                                       while (abs(tmp_yaw - imu_data.yaw) > 9000) {
                                           if (tmp_yaw - imu_data.yaw >= 9000)
                                               tmp_yaw -= 18000;
                                           else
                                               tmp_yaw += 18000;
                                       }
                                       SerialParam::send_data.yaw = tmp_yaw;


                                       DLOG(INFO) << "                                        send yaw: " << SerialParam::send_data.yaw
                                                  << "  send pitch: " << SerialParam::send_data.pitch;
                                       DLOG(INFO) << "                                        recv yaw: " << SerialParam::recv_data.yaw
                                                  << "  recv pitch: " << SerialParam::recv_data.pitch;

                                       last_armor = armor; // 上一次识别到的armor
                                       return reproject(shootCenter);
                                   } else {
                                       getCenter = false;
                                       // 掉帧缓冲
                                       if (lost_cnt < 20)
                                           lost_cnt++;
                                       else {
                                           lost_cnt = 0;
//                                           predictor->resetPredictor();
                                       }

                                       last_armor = armor; // 上一次识别到的armor
                                       return cv::Point2f(0, 0);
                                   }

        return cv::Point2f(0,0);
}




