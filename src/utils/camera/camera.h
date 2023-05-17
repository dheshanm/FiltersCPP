// SPDX-FileCopyrightText: 2023 Dheshan Mohandass (L4TTiCe)
//
// SPDX-License-Identifier: MIT

#ifndef VISION_CPP_CAMERA_H
#define VISION_CPP_CAMERA_H

#include <opencv2/opencv.hpp>

/**
 * A class that represents a camera device and provides methods to capture frames from it.
 * The camera is opened using an index that corresponds to the device ID or a video file name.
 * The camera can be configured to have a certain frame rate using the set_fps method.
 * The camera can be used to read frames into a cv::Mat object using the read method.
 * The camera is automatically closed when the object is destroyed.
 */
class Camera {
public:
    /**
    * A constructor that creates a Camera object and opens the camera device with the given index.
    * @param index an integer that specifies the camera device ID or a video file name
    */
    explicit Camera(int index);

    /**
    * A destructor that releases the camera device and frees any resources associated with it.
    */
    ~Camera();

    /**
    * A method that sets the frame rate of the camera device in frames per second.
    * @param fps an integer that specifies the desired frame rate
    */
    void set_fps(int fps);

    /**
    * A method that reads a frame from the camera device and stores it in a cv::Mat object.
    * @param frame a reference to a cv::Mat object where the captured frame will be stored
    * @return 0 if the frame was successfully captured, -1 otherwise
    */
    int read(cv::Mat &frame);

private:
    cv::VideoCapture videoCapture; // a cv::VideoCapture object that represents the camera device
    [[maybe_unused]] int index; // an integer that stores the index of the camera device
};


#endif //VISION_CPP_CAMERA_H
