//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_KERNELS_H
#define VISION_CPP_KERNELS_H

#include <opencv2/opencv.hpp>

/**
 * Returns a valid index for accessing an array or matrix element, given an index, an offset and a maximum value.
 * The function ensures that the result is within the range [0, max) by wrapping around the boundaries.
 * For example, if index = 0, offset = -1 and max = 10, the result is 9. If index = 9, offset = 1 and max = 10, the result is 0.
 * @param index The original index of the element.
 * @param offset The offset to be added to the index.
 * @param max The maximum value of the index (exclusive).
 * @return A valid index that is within the range [0, max).
 */
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

/**
 * Applies a partial kernel to a row of an input image and stores the result in an output image.
 * The partial kernel is a one-dimensional vector of integers that represents a convolution filter.
 * The function performs a weighted sum of the pixel values in the row and its neighboring rows, using the kernel values as weights.
 * The function also normalizes the result by dividing it by the sum of the kernel values, or by 1 if the sum is zero.
 * The function uses OpenMP directives to parallelize the computation for each row.
 * @param input The input image (a matrix of 3-channel pixels).
 * @param output The output image (a matrix of 3-channel pixels).
 * @param kernel The partial kernel (a vector of integers).
 * @param kernel_offset The offset of the kernel from the center of the row. For example, if kernel_offset = 1, then the kernel is applied to the row and its upper neighbor. If kernel_offset = 2, then the kernel is applied to the row and its upper and upper-upper neighbors.
 */
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

/**
 * Applies a partial kernel to a column of an input image and stores the result in an output image.
 * The partial kernel is a one-dimensional vector of integers that represents a convolution filter.
 * The function performs a weighted sum of the pixel values in the column and its neighboring columns, using the kernel values as weights.
 * The function also normalizes the result by dividing it by the sum of the kernel values, or by 1 if the sum is zero.
 * The function uses OpenMP directives to parallelize the computation for each column.
 * @param input The input image (a matrix of 3-channel pixels).
 * @param output The output image (a matrix of 3-channel pixels).
 * @param kernel The partial kernel (a vector of integers).
 * @param kernel_offset The offset of the kernel from the center of the column. For example, if kernel_offset = 1, then the kernel is applied to the column and its left neighbor. If kernel_offset = 2, then the kernel is applied to the column and its left and left-left neighbors.
 */
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

/**
 * Applies a full kernel to an input image and stores the result in an output image.
 * The kernel is a two-dimensional matrix of integers that represents a convolution filter.
 * The function performs a weighted sum of the pixel values in the image and its neighboring pixels, using the kernel values as weights.
 * The function also normalizes the result by dividing it by the sum of the kernel values, or by 1 if the sum is zero.
 * The function uses OpenMP directives to parallelize the computation for each row.
 *
 * This function used a partial kernel to apply the kernel to each row, and then uses the same partial kernel to apply the kernel to each column.
 *
 * @param input The input image (a matrix of 3-channel pixels).
 * @param output The output image (a matrix of 3-channel pixels).
 * @param kernel The kernel to be used for both rows and columns (a vector of integers).
 * @param kernel_offset The offset of the kernel from the center of each pixel. For example, if kernel_offset = 1, then the kernel is a 3x3 matrix. If kernel_offset = 2, then the kernel is a 5x5 matrix.
 */
void apply_kernel(cv::Mat& input, cv::Mat& output, std::vector<int>& kernel, int kernel_offset) {
    cv::Mat intermediate = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    apply_partial_kernel_row(input, intermediate, kernel, kernel_offset);
    apply_partial_kernel_col(intermediate, output, kernel, kernel_offset);
}

#endif //VISION_CPP_KERNELS_H
