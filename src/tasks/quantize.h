//
// Created by master on 5/15/23.
//

#ifndef VISION_CPP_QUANTIZE_H
#define VISION_CPP_QUANTIZE_H

#include <opencv2/opencv.hpp>
#include <thread>
#include "../utils/watch_channel.h"
#include "../utils/processor/processor.h"
#include "../utils/filters.h"
#include "task.h"
#include "../constants.h"

void quantize_task(WatchChannel<cv::Mat>& inputChannel, WatchChannel<cv::Mat>& outputChannel) {
    cv::Mat frame;
    inputChannel.read(frame);
    if (frame.empty()) {
        return;
    }

    cv::Mat output_frame;
    quantize(frame, output_frame, 15);
    outputChannel.write(output_frame);
}

void quantize_process(WatchChannel<cv::Mat>& inputChannel, WatchChannel<cv::Mat>& outputChannel, ProcessorState& processorState) {
    Processor processor("Quantize", &processorState);

    processor.register_callback(quantize_task);
    processor.start(inputChannel, outputChannel);
}

class QuantizedTask : public Task {
public:
    explicit QuantizedTask(WatchChannel<cv::Mat> &outputChannel) : Task(QUANTIZED, outputChannel) {}

    void start(WatchChannel<cv::Mat>& input) {
        processorThread = std::thread(quantize_process, std::ref(input), std::ref(*outputChannel), std::ref(processorState));
    }
};

#endif //VISION_CPP_QUANTIZE_H
