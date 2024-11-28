#ifndef BUMPER_H
#define BUMPER_H

#include<string>

#define MAX_LOSS_BUMP_COUNT 35
#define MAX_DETECT_BUMP_COUNT 5
    enum DETECT_MODE
    {
        NOT_GET_TARGET=0,       // 未获得目标
        CONTINOUS_GET_TARGET=1, // 连续获得目标
        LOST_BUMP=2,            // 缓冲阶段
        DETECT_BUMP=3           // 进入连续识别状态的缓冲

    };
    class Bumper
    {
    public:
        Bumper()=default;
        ~Bumper()=default;

        // 获取检测模式
        int getDetectMode(int &target_class,int &last_frame_class)
        {
            // 初始没有目标    ---检测到--》  检测缓冲状态

            // 当前无目标
            if(mode==NOT_GET_TARGET)
            {
                if(target_class<=0)     // 丢失目标
                {
                    target_loss_count++;
                }
                else if(target_class>0)
                {
                    target_loss_count=0;
                    mode=CONTINOUS_GET_TARGET;   //连续识别状态
                }
                if(target_loss_count>MAX_LOSS_BUMP_COUNT)   // 连续识别不到次数
                {
                    mode =NOT_GET_TARGET;
                    target_loss_count=0;
                }
            }
            // 连续识别状态  ---丢失帧数--->   掉帧缓冲状态

            // 连续识别状态
            else if(mode==CONTINOUS_GET_TARGET)
            {
                // 进入调帧缓冲
                if(target_class<=0||target_class!=last_frame_class)
                {
                    mode=LOST_BUMP;
                }
            }
            return mode;
        }
    private:
        int target_loss_count=0;
        int detect_bump_count=0;
        int mode=NOT_GET_TARGET;
    };

#endif // BUMPER_H
