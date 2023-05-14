//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_CAMERA_H
#define VISION_CPP_CAMERA_H

#include <opencv2/opencv.hpp>

class Camera {
public:
    explicit Camera(int index);
    ~Camera();
    void set_fps(int fps);
    int read(cv::Mat& frame);
private:
    cv::VideoCapture videoCapture;
    int index;
};


#endif //VISION_CPP_CAMERA_H
