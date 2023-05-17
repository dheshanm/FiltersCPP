// SPDX-FileCopyrightText: 2023 Dheshan Mohandass (L4TTiCe)
//
// SPDX-License-Identifier: MIT

#ifndef VISION_CPP_GREYSCALE_H
#define VISION_CPP_GREYSCALE_H

#include <opencv2/opencv.hpp>
#include "../utils/processor/processor.h"
#include "../utils/filters.h"
#include "task.h"
#include "../constants.h"

void grayscale_task(WatchChannel<cv::Mat> &inputChannel, WatchChannel<cv::Mat> &outputChannel) {
    cv::Mat frame;
    inputChannel.read(frame);
    if (frame.empty()) {
        return;
    }
    cv::Mat grayscale_frame;
    grayscale(frame, grayscale_frame);
    outputChannel.write(grayscale_frame);
}

void grayscale_process(WatchChannel<cv::Mat> &inputChannel, WatchChannel<cv::Mat> &outputChannel,
                       ProcessorState &processorState) {
    Processor processor("Grayscale", &processorState);

    processor.register_callback(grayscale_task);
    processor.start(inputChannel, outputChannel);
}

class GrayscaleTask : public Task {
public:
    explicit GrayscaleTask(WatchChannel<cv::Mat> &outputChannel) : Task(GRAYSCALE, outputChannel) {}

    void start(WatchChannel<cv::Mat> &input) {
        processorThread = std::thread(grayscale_process, std::ref(input), std::ref(*outputChannel),
                                      std::ref(processorState));
    }
};

#endif //VISION_CPP_GREYSCALE_H
