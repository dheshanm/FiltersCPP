//
// Created by master on 5/14/23.
//

#include "camera.h"

Camera::Camera(int index) {
    this->index = index;
    videoCapture.open(index);
    if (!videoCapture.isOpened()) {
        std::cout << "Failed to open camera." << std::endl;
        return;
    }
}

Camera::~Camera() {
    videoCapture.release();
}

void Camera::set_fps(int fps) {
    videoCapture.set(cv::CAP_PROP_FPS, fps);
}

int Camera::read(cv::Mat &frame) {
    videoCapture >> frame;
    if (frame.empty()) {
        std::cout << "Failed to capture frame." << std::endl;
        return -1;
    }
    return 0;
}
