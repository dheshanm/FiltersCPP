//
// Created by master on 5/15/23.
//

#ifndef VISION_CPP_CARTOONIZE_H
#define VISION_CPP_CARTOONIZE_H

#include <opencv2/opencv.hpp>
#include <thread>
#include "../utils/watch_channel.h"
#include "../utils/processor/processor.h"
#include "../utils/filters.h"
#include "task.h"
#include "../constants.h"

void cartoonize_task(WatchChannel<cv::Mat>& quantized_input, WatchChannel<cv::Mat>& magnitude_input, WatchChannel<cv::Mat>& output_channel) {
    cv::Mat quantized_frame, magnitude_frame;

    quantized_input.read(quantized_frame);
    magnitude_input.read(magnitude_frame);

    if (quantized_frame.empty() || magnitude_frame.empty()) {
        return;
    }
    cv::Mat output_frame;
    cartoonize(quantized_frame, magnitude_frame, output_frame, 20);
    output_channel.write(output_frame);
}

void cartoonize_process(WatchChannel<cv::Mat>& input_channel_1, WatchChannel<cv::Mat>& input_channel_2, WatchChannel<cv::Mat>& output_channel, ProcessorState& processorState) {
    DualInputProcessor processor("", &processorState);

    processor.register_callback(cartoonize_task);
    processor.start(input_channel_1, input_channel_2, output_channel);
}

class CartoonizeTask : public Task {
public:
    explicit CartoonizeTask(WatchChannel<cv::Mat> &outputChannel) : Task(CARTOONIZE, outputChannel) {}

    void start(WatchChannel<cv::Mat>& quantized_input, WatchChannel<cv::Mat>& magnitude_input) {
        processorThread = std::thread(cartoonize_process, std::ref(quantized_input), std::ref(magnitude_input), std::ref(*outputChannel), std::ref(processorState));
    }
};


#endif //VISION_CPP_CARTOONIZE_H
