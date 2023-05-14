//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_KERNELS_H
#define VISION_CPP_KERNELS_H

#include <opencv2/opencv.hpp>

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

void apply_partial_kernel_row(cv::Mat& input, cv::Mat& output, std::vector<int>& kernel, int kernel_offset) {
    int kernel_sum = 0;
    for (int i : kernel) {
        kernel_sum += i;
    }
    if (kernel_sum == 0) {
        kernel_sum = 1;
    }

#pragma omp parallel for default(none) shared(kernel_offset, input, output, kernel, kernel_sum)
    for (int row = 0; row < input.rows; row++) {
        for (int col = 0; col < input.cols; col++) {
            cv::Vec3i buffer_result = cv::Vec3i {0, 0, 0};

            int kernel_idx = 0;
            for (int row_offset = -kernel_offset; row_offset <= kernel_offset; row_offset++) {
                int row_idx = get_valid_index(row, row_offset, input.rows);

                cv::Vec3b current_pixel = input.at<cv::Vec3b>(row_idx, col);

                for (int channel_idx = 0; channel_idx < 3; channel_idx++) {
                    buffer_result[channel_idx] += current_pixel[channel_idx] * kernel[kernel_idx];
                }
                kernel_idx++;
            }

            cv::Vec3b pixel = buffer_result / kernel_sum;
            output.at<cv::Vec3b>(row, col) = pixel;
        }
    }
}

void apply_partial_kernel_col(cv::Mat& input, cv::Mat& output, std::vector<int>& kernel, int kernel_offset) {
    int kernel_sum = 0;
    for (int i: kernel) {
        kernel_sum += i;
    }
    if (kernel_sum == 0) {
        kernel_sum = 1;
    }

#pragma omp parallel for default(none) shared(kernel_offset, input, output, kernel, kernel_sum)
    for (int row = 0; row < input.rows; row++) {
        for (int col = 0; col < input.cols; col++) {
            cv::Vec3i buffer_result = cv::Vec3i {0, 0, 0};
            int kernel_idx = 0;

            for (int col_offset = -kernel_offset; col_offset <= kernel_offset; col_offset++) {
                int col_idx = get_valid_index(col, col_offset, input.cols);

                cv::Vec3b current_pixel = input.at<cv::Vec3b>(row, col_idx);
                for (int channel_idx = 0; channel_idx < 3; channel_idx++) {
                    buffer_result[channel_idx] += current_pixel[channel_idx] * kernel[kernel_idx];
                }
                kernel_idx++;
            }

            cv::Vec3b pixel = buffer_result/ kernel_sum;
            output.at<cv::Vec3b>(row, col) = pixel;
        }
    }
}

void apply_kernel(cv::Mat& input, cv::Mat& output, std::vector<int>& kernel, int kernel_offset) {
    cv::Mat intermediate = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    apply_partial_kernel_row(input, intermediate, kernel, kernel_offset);
    apply_partial_kernel_col(intermediate, output, kernel, kernel_offset);
}

#endif //VISION_CPP_KERNELS_H
