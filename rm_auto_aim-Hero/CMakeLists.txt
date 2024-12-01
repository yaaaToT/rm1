cmake_minimum_required(VERSION 3.1)
project(rm_auto_aim-Hero)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Sophus REQUIRED)

#find_library(mvsdk libMVSDK.so HINTS /usr/lib)
#find_library(thread libpthread.so.0 HINTS /lib/x86_64-linux-gnu)



include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(rm_auto_aim-Hero main.cpp
        include/armor_detector/armor.h
        include/armor_detector/ArmorDetetion.h
        #include/Energy/Energy.h
        include/kalmanfilter/KalmanFilter.h
        #include/mindvision/camera_mindvision_running.h
        #include/mindvision/CameraApi.h
        #include/mindvision/CameraDefine.h
        #include/mindvision/CameraStatus.h
        include/Preprocess/Preprocess.h
        include/Resolver/GimbalControl.h
        include/Resolver/Resolver.h
        include/SerialPort/SerialPort.h
        include/outpost/outpost.h
        include/outpost/Array.h
        include/outpost/newArray.h
        include/Tracker/Tracker.h
        include/debug.h
        src/armor_detector/ArmorDetetion.cpp
        src/armor_detector/armor.cpp
        #src/Energy/Energy.cpp
        #src/mindvision/camera_mindvision_running.cpp
        src/kalmanfilter/KalmanFilter.cpp
        src/Resolver/GimbalControl.cpp
        src/Resolver/Resolver.cpp
        src/SerialPort/SerialPort.cpp
        src/Tracker/Tracker.cpp
        src/outpost/outpost.cpp
        )


target_link_libraries(rm_auto_aim-Hero
        ${OpenCV_LIBS}
        ${mvsdk}
        ${thread}
        ${Sophus_LIBRARIES}
)
