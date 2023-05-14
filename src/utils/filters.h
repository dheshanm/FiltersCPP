//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_FILTERS_H
#define VISION_CPP_FILTERS_H

#include <opencv2/opencv.hpp>

void grayscale(cv::Mat& frame, cv::Mat& output) {
    cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
}

void negative(cv::Mat& input, cv::Mat& output) {
    cv::bitwise_not(input, output);
}

int get_valid_index(int index, int offset, int max) {
    int result = index + offset;
    if (result < 0) {
        return -result;
    } else if (result >= max) {
        return max - (result - max) - 1;
    } else {
        return result;
    }
}

void blur5x5(cv::Mat& input, cv::Mat& output) {
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    std::vector<int> kernel = {1, 2, 4, 2, 1};
    int kernel_sum = 0;
    for (int i : kernel) {
        kernel_sum += i;
    }

    #pragma omp parallel for default(none) shared(input, output, kernel, kernel_sum)
    for (int row = 0; row < input.rows; row++) {
        for (int col = 0; col < input.cols; col++) {
            cv::Mat product = cv::Mat::zeros(1, kernel.size(), CV_32SC3);

            int product_idx = 0;
            for (int i = -2; i <= 2; i++) {
                int row_idx = get_valid_index(row, i, input.rows);
                int kernel_idx = 0;

                cv::Vec3i buffer_result = cv::Vec3i {0, 0, 0};

                for (int j = -2; j <= 2; j++) {
                    int col_idx = get_valid_index(col, j, input.cols);

                    cv::Vec3b current_pixel = input.at<cv::Vec3b>(row_idx, col_idx);

                    buffer_result += current_pixel * kernel[kernel_idx];
                    kernel_idx++;
                }

                product.at<cv::Vec3i>(0, product_idx) = buffer_result;
                product_idx++;
            }

            cv::Vec3b pixel = cv::Vec3b {
                    static_cast<uchar>(product.at<cv::Vec3i>(0, 0)[0] / kernel_sum),
                    static_cast<uchar>(product.at<cv::Vec3i>(0, 1)[1] / kernel_sum),
                    static_cast<uchar>(product.at<cv::Vec3i>(0, 2)[2] / kernel_sum)
            };

            output.at<cv::Vec3b>(row, col) = pixel;
        }
    }
}

#endif //VISION_CPP_FILTERS_H
