//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_SOBEL_H
#define VISION_CPP_SOBEL_H

#include <opencv2/opencv.hpp>
#include "../utils/watch_channel.h"
#include "../utils/processor/processor.h"
#include "../utils/filters.h"

void sobel_x_task(WatchChannel<cv::Mat>& inputChannel, WatchChannel<cv::Mat>& outputChannel) {
    cv::Mat frame;
    inputChannel.read(frame);
    if (frame.empty()) {
        return;
    }
    cv::Mat output;
    sobel_x(frame, output);
    outputChannel.write(output);
}

void sobel_x_process(WatchChannel<cv::Mat>& inputChannel, WatchChannel<cv::Mat>& outputChannel, ProcessorState& processorState) {
    Processor processor("Sobel X", &processorState);

    processor.register_callback(sobel_x_task);
    processor.start(inputChannel, outputChannel);
}

void sobel_y_task(WatchChannel<cv::Mat>& inputChannel, WatchChannel<cv::Mat>& outputChannel) {
    cv::Mat frame;
    inputChannel.read(frame);
    if (frame.empty()) {
        return;
    }
    cv::Mat output;
    sobel_y(frame, output);
    outputChannel.write(output);
}

void sobel_y_process(WatchChannel<cv::Mat>& inputChannel, WatchChannel<cv::Mat>& outputChannel, ProcessorState& processorState) {
    Processor processor("Sobel Y", &processorState);

    processor.register_callback(sobel_y_task);
    processor.start(inputChannel, outputChannel);
}

#endif //VISION_CPP_SOBEL_H
