
#ifndef SERIALPORT_H
#define SERIALPORT_H

#include "stdint.h"
#include <errno.h>  /*错误号定义*/
#include <fcntl.h>  /*文件控制定义*/
#include <stdio.h>  /*标准输入输出定义*/
#include <stdlib.h> /*标准函数库定义*/
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <termios.h> /*PPSIX 终端控制定义*/
#include <unistd.h>  /*Unix 标准函数定义*/
#include <iostream>

#define BAUDRATE 115200
#define UART_DEVICE "/dev/ttyACM0"

#define ECHOFLAGS (ECHO | ECHOE | ECHOK | ECHONL)

// Used when calling a serial port
#define FALSE -1
#define TRUE 0
#define TIMEOUT_SEC(buflen, baud) (buflen * 20 / baud + 2)  //接收超时
#define TIMEOUT_USEC 0

typedef union {
    float         f;
    unsigned char c[4];
} float2uchar;

typedef union {
    unsigned char c[4];
    float         f;
} char2float;

typedef union {
    char    c[2];
    int16_t f;
} uint2int16;

//fasong
struct HostComputerData
{
    float2uchar   Pitch;
    float2uchar   Yaw;
    float2uchar   distance;

    int yaw;
    short pitch;

    unsigned char if_shoot;
    unsigned char shootStatus;
            int time_stamp;
            unsigned char state;
            unsigned char num;
            unsigned char crc;
    unsigned char if_real_shoot;

    HostComputerData(){

    }
    HostComputerData(const HostComputerData&pdata){

    }


} ;

//jieshou
struct GroundChassisData
{
    float2uchar   gain_pitch;
    float2uchar   gain_yaw;

    float2uchar   gain_roll;

    int yaw;
    short pitch;
    short roll;

    unsigned char mode;
    unsigned char color;  // unsigned char
    unsigned char speed;
    unsigned char zaishimeiyong;

    unsigned char right_clicked;
            char user_time_bias;
            char outpost_state;
            unsigned char crc;

            GroundChassisData(){

            }
            GroundChassisData(const GroundChassisData&data){
                this->yaw=data.yaw;
                this->pitch=data.pitch;
                this->roll=data.roll;
            }

    // unsigned char selfid;
} ;

extern HostComputerData  RobotInfo;        // 发送数据
extern GroundChassisData MainControlInfo;  // 接收数据

class Uart
{
public:

    explicit Uart();
    explicit Uart(const std::string& port_name);
    ~Uart();

    Uart(int speed = 9600);
    int     Init_serial(int& fdcom, int baud_speed);
    void    set_speed(int fd, speed_t speed);
    int     set_Parity(int fd, int databits, int stopbits, int parity);
    int     send_data(int fd, unsigned char* buffer, int Length);
    int     received_data(int fd, unsigned char* buffer, int datalen, int baudrate);
    void    close_serial(int fd);
    void    GetMode(int& fd, GroundChassisData& data);
    void    TransformTarPos(int& fd, const HostComputerData& data);

    void    Recdata(GroundChassisData*imu_data);
    void    Inpdata(HostComputerData*_data_write);

    uint8_t crc8_check(uint8_t* addr, int len);

private:
    int fd;
    int speed;
};

extern Uart InfoPort;

#endif //SERIALPORT_H
