// SPDX-FileCopyrightText: 2023 Dheshan Mohandass (L4TTiCe)
//
// SPDX-License-Identifier: MIT

#ifndef VISION_CPP_FILTERS_H
#define VISION_CPP_FILTERS_H

#include <opencv2/opencv.hpp>
#include "kernels.h"

/**
 * This function converts a color image to grayscale using OpenCV library
 * It takes two parameters: frame (the input image) and output (the output image)
 * It does not return anything
 * @param frame The input color image
 * @param output The output grayscale image
 */
void grayscale(cv::Mat &frame, cv::Mat &output) {
    cv::cvtColor(frame, output, cv::COLOR_BGR2GRAY);
}

/**
 * This function creates a negative image by inverting the pixel values using OpenCV library
 * It takes two parameters: input (the input image) and output (the output image)
 * It does not return anything
 * @param input The input image
 * @param output The output negative image
 */
void negative(cv::Mat &input, cv::Mat &output) {
    cv::bitwise_not(input, output);
}

/**
 * This function applies a 5x5 blur filter to an image using a custom kernel
 * It takes three parameters: input (the input image), output (the output image), and kernel (a vector of integers representing the filter coefficients)
 * It does not return anything
 * It calls the apply_kernel function to perform the convolution operation
 * @param input The input image
 * @param output The output blurred image
 * @param kernel The vector of filter coefficients
 */
void blur5x5(cv::Mat &input, cv::Mat &output) {
    std::vector<int> kernel = {2, 4, 6, 4, 2};
    apply_kernel(input, output, kernel, 2);
}

/**
 * This function computes the horizontal gradient of an image using Sobel operator
 * It takes two parameters: input (the input image) and output (the output image)
 * It does not return anything
 * It uses two 1D kernels to perform the convolution operation in two steps: first along the rows and then along the columns
 * It calls the apply_partial_kernel_row and apply_partial_kernel_col functions to perform the partial convolutions
 * @param input The input image
 * @param output The output horizontal gradient image
 */
void sobel_x(cv::Mat &input, cv::Mat &output) {
    std::vector<int> kernel_1 = {1, 2, 1};
    std::vector<int> kernel_2 = {-1, 0, +1};

//    cv::Mat intermediate = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

//    apply_partial_kernel_row(input, intermediate, kernel_1, 1);
    apply_partial_kernel_col(input, output, kernel_2, 1);
}

/**
 * This function computes the vertical gradient of an image using Sobel operator
 * It takes two parameters: input (the input image) and output (the output image)
 * It does not return anything
 * It uses two 1D kernels to perform the convolution operation in two steps: first along the rows and then along the columns
 * It calls the apply_partial_kernel_row and apply_partial_kernel_col functions to perform the partial convolutions
 * @param input The input image
 * @param output The output vertical gradient image
 */
void sobel_y(cv::Mat &input, cv::Mat &output) {
    std::vector<int> kernel_1 = {-1, 0, +1};
    std::vector<int> kernel_2 = {1, 2, 1};

//    cv::Mat intermediate = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);
    output = cv::Mat::zeros(input.rows, input.cols, CV_8UC3);

    apply_partial_kernel_row(input, output, kernel_1, 1);
//    apply_partial_kernel_col(intermediate, output, kernel_2, 1);
}

/**
 * This function computes the magnitude of the gradient of an image using the outputs of sobel_x and sobel_y functions
 * It takes three parameters: sobel_input_1 (the horizontal gradient image), sobel_input_2 (the vertical gradient image), and output (the output image)
 * It does not return anything
 * It throws an exception if the inputs are not of the same size
 * It uses OpenMP to parallelize the computation for each pixel
 * @param sobel_input_1 The horizontal gradient image
 * @param sobel_input_2 The vertical gradient image
 * @param output The output magnitude of the gradient image
 */
void magnitude(cv::Mat &sobel_input_1, cv::Mat &sobel_input_2, cv::Mat &output) {
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

/**
 * This function quantizes an image into a given number of levels using OpenCV library
 * It takes four parameters: input (the input image), output (the output image), levels (an integer representing the number of levels), and blur (a boolean indicating whether to blur the image before quantization or not)
 * It does not return anything
 * It throws an exception if the levels are less than 2
 * It uses OpenMP to parallelize the computation for each pixel
 * @param input The input image
 * @param output The output quantized image
 * @param levels The number of levels for quantization
 * @param blur A flag indicating whether to blur the image or not before quantization. Default is true.
 */
void quantize(cv::Mat &input, cv::Mat &output, int levels, bool blur = true) {
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

/**
 * This function creates a cartoon-like effect on an image by combining quantization and edge detection
 * It takes four parameters: quantized_input (the quantized image), magnitude_input (the magnitude of the gradient image), output (the output image), and magnitude_threshold (an integer representing the threshold for edge detection)
 * It does not return anything
 * It throws an exception if the inputs are not of the same size
 * It uses OpenMP to parallelize the computation for each pixel
 * @param quantized_input The quantized image
 * @param magnitude_input The magnitude of the gradient image
 * @param output The output cartoonized image
 * @param magnitude_threshold The threshold for edge detection
 */
void cartoonize(cv::Mat &quantized_input, cv::Mat &magnitude_input, cv::Mat &output, int magnitude_threshold) {
    if (quantized_input.rows != magnitude_input.rows || quantized_input.cols != magnitude_input.cols) {
        throw std::invalid_argument("Inputs must be the same size");
    }

    output = cv::Mat::zeros(quantized_input.rows, quantized_input.cols, CV_8UC3);

# pragma omp parallel for default(none) shared(quantized_input, magnitude_input, output, magnitude_threshold)
    for (int row_idx = 0; row_idx < quantized_input.rows; row_idx++) {
        for (int col_idx = 0; col_idx < quantized_input.cols; col_idx++) {
            cv::Vec3b quantized_pixel = quantized_input.at<cv::Vec3b>(row_idx, col_idx);
            cv::Vec3b magnitude_pixel = magnitude_input.at<cv::Vec3b>(row_idx, col_idx);

            int magnitude = magnitude_pixel[0];
            if (magnitude > magnitude_threshold) {
                output.at<cv::Vec3b>(row_idx, col_idx) = cv::Vec3b(0, 0, 0);
            } else {
                output.at<cv::Vec3b>(row_idx, col_idx) = quantized_pixel;
            }
        }
    }
}

#endif //VISION_CPP_FILTERS_H
