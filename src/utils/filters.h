//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_FILTERS_H
#define VISION_CPP_FILTERS_H

#include <opencv2/opencv.hpp>
#include "kernels.h"

void grayscale(cv::Mat& frame, cv::Mat& output) {
    cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
}

void negative(cv::Mat& input, cv::Mat& output) {
    cv::bitwise_not(input, output);
}

void blur5x5(cv::Mat& input, cv::Mat& output) {
    std::vector<int> kernel = {2, 4, 6, 4, 2};
    apply_kernel(input, output, kernel, 2);
}

void sobel_x(cv::Mat& input, cv::Mat& output) {
    std::vector<int> kernel_1 = {1, 2, 1};
    std::vector<int> kernel_2 = {-1, 0, +1};

    cv::Mat intermediate = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    apply_partial_kernel_row(input, intermediate, kernel_1, 1);
    apply_partial_kernel_col(intermediate, output, kernel_2, 1);
}

void sobel_y(cv::Mat& input, cv::Mat& output) {
    std::vector<int> kernel_1 = {-1, 0, +1};
    std::vector<int> kernel_2 = {1, 2, 1};

    cv::Mat intermediate = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    apply_partial_kernel_row(input, intermediate, kernel_1, 1);
    apply_partial_kernel_col(intermediate, output, kernel_2, 1);
}

#endif //VISION_CPP_FILTERS_H
