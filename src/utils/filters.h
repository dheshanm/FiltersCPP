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

void magnitude(cv::Mat& sobel_input_1, cv::Mat& sobel_input_2, cv::Mat& output) {
    if (sobel_input_1.rows != sobel_input_2.rows || sobel_input_1.cols != sobel_input_2.cols) {
        throw std::invalid_argument("Sobel inputs must be the same size");
    }

    output = cv::Mat::zeros(sobel_input_1.rows, sobel_input_1.cols, CV_8UC3);

    # pragma omp parallel for default(none) shared(sobel_input_1, sobel_input_2, output)
    for (int row_idx = 0; row_idx < sobel_input_1.rows; row_idx++) {
        for (int col_idx = 0; col_idx < sobel_input_1.cols; col_idx++) {
            cv::Vec3b pixel_1 = sobel_input_1.at<cv::Vec3b>(row_idx, col_idx);
            cv::Vec3b pixel_2 = sobel_input_2.at<cv::Vec3b>(row_idx, col_idx);

            int magnitude = std::sqrt(std::pow(pixel_1[0], 2) + std::pow(pixel_2[0], 2));
            output.at<cv::Vec3b>(row_idx, col_idx) = cv::Vec3b(magnitude, magnitude, magnitude);
        }
    }
}

void quantize(cv::Mat& input, cv::Mat& output, int levels, bool blur = true) {
    if (levels < 2) {
        throw std::invalid_argument("Levels must be greater than 1");
    }

    int bins_count = 255 / levels;

    // blur the image to reduce noise
    if (blur) {
        cv::Mat blurred = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
        cv::GaussianBlur(input, blurred, cv::Size(5, 5), 0);
        input = blurred;
    }

    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    # pragma omp parallel for default(none) shared(input, output, bins_count)
    for (int row_idx = 0; row_idx < input.rows; row_idx++) {
        for (int col_idx = 0; col_idx < input.cols; col_idx++) {
            cv::Vec3b pixel = input.at<cv::Vec3b>(row_idx, col_idx);
            for (int channel_idx = 0; channel_idx < 3; channel_idx++) {
                int quantized = std::round(pixel[channel_idx] / bins_count) * bins_count;
                output.at<cv::Vec3b>(row_idx, col_idx)[channel_idx] = quantized;
            }
        }
    }
}

#endif //VISION_CPP_FILTERS_H
