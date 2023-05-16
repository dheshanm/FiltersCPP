//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_NEGATIVE_H
#define VISION_CPP_NEGATIVE_H

#include <opencv2/opencv.hpp>
#include "../utils/watch_channel.h"
#include "../utils/processor/processor.h"
#include "../utils/filters.h"

void negative_task(WatchChannel<cv::Mat> &inputChannel, WatchChannel<cv::Mat> &outputChannel) {
    cv::Mat frame;
    inputChannel.read(frame);
    if (frame.empty()) {
        return;
    }

    cv::Mat negative_frame;
    negative(frame, negative_frame);
    outputChannel.write(negative_frame);
}

void negative_process(WatchChannel<cv::Mat> &inputChannel, WatchChannel<cv::Mat> &outputChannel,
                      ProcessorState &processorState) {
    Processor processor("Negative", &processorState);

    processor.register_callback(negative_task);
    processor.start(inputChannel, outputChannel);
}

class NegativeTask : public Task {
public:
    explicit NegativeTask(WatchChannel<cv::Mat> &outputChannel) : Task(NEGATIVE, outputChannel) {}

    void start(WatchChannel<cv::Mat> &input) {
        processorThread = std::thread(negative_process, std::ref(input), std::ref(*outputChannel),
                                      std::ref(processorState));
    }
};

#endif //VISION_CPP_NEGATIVE_H
