//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_BLUR_H
#define VISION_CPP_BLUR_H

#include <opencv2/opencv.hpp>
#include "../utils/watch_channel.h"
#include "../utils/filters.h"
#include "../utils/processor/processor.h"

void blur_task(WatchChannel<cv::Mat>& inputChannel, WatchChannel<cv::Mat>& outputChannel) {
    cv::Mat frame;
    inputChannel.read(frame);
    if (frame.empty()) {
        return;
    }

    cv::Mat blur_frame;
    blur5x5(frame, blur_frame);
    outputChannel.write(blur_frame);
}

void blur_process(WatchChannel<cv::Mat>& inputChannel, WatchChannel<cv::Mat>& outputChannel, ProcessorState& processorState) {
    Processor processor("Blur", &processorState);

    processor.register_callback(blur_task);
    processor.start(inputChannel, outputChannel);
}

#endif //VISION_CPP_BLUR_H
