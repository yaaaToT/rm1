cmake_minimum_required(VERSION 3.1)
project(rm-hero)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Sophus REQUIRED)

#find_library(mvsdk libMVSDK.so HINTS /usr/lib)
#find_library(thread libpthread.so.0 HINTS /lib/x86_64-linux-gnu)



include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")


add_executable(rm-hero main.cpp
        include/armor_detector/armor.h
        include/armor_detector/ArmorDetetion.h
        include/armor_detector/TargetChooser.h
        include/armor_detector/classifier.h
        include/armor_detector/inference.h
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
        include/SerialPort/UDPSender.h
        include/outpost/Array.h
        include/outpost/newArray.h
        include/outpost/params.h
        include/outpost/outpost.h
        include/Tracker/Tracker.h
        include/Tracker/Predictor.h
        include/Tracker/bumper.h
        include/debug.h
        src/armor_detector/ArmorDetetion.cpp
        src/armor_detector/armor.cpp
        src/armor_detector/TargetChooser.cpp
        src/armor_detector/inference.cpp
        #src/Energy/Energy.cpp
        #src/mindvision/camera_mindvision_running.cpp
        src/kalmanfilter/KalmanFilter.cpp
        src/Resolver/GimbalControl.cpp
        src/Resolver/Resolver.cpp
        src/SerialPort/SerialPort.cpp
        src/Tracker/Tracker.cpp
        src/Tracker/Predictor.cpp
        src/outpost/outpost.cpp
        src/outpost/params.cpp
        )

include_directories(
            ${OpenCV_INCLUDE_DIRS}
            /opt/intel/openvino_2024/runtime/include

    )

    target_link_libraries(${PROJECT_NAME}
           ${OpenCV_LIBS}
           /opt/intel/openvino_2024/runtime/lib/intel64/libopenvino.so
    )


target_link_libraries(rm-hero
        ${OpenCV_LIBS}
        ${mvsdk}
        ${thread}
        ${Sophus_LIBRARIES}
)
