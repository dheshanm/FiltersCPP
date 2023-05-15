//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_MAGNITUDE_H
#define VISION_CPP_MAGNITUDE_H

#include <opencv2/opencv.hpp>
#include "../utils/watch_channel.h"
#include "../utils/processor/processor.h"

void magnitude_task(WatchChannel<cv::Mat>& input_channel_1, WatchChannel<cv::Mat>& input_channel_2, WatchChannel<cv::Mat>& output_channel) {
    cv::Mat input_frame_1, input_frame_2;

    input_channel_1.read(input_frame_1);
    input_channel_2.read(input_frame_2);

    if (input_frame_1.empty() || input_frame_2.empty()) {
        return;
    }
    cv::Mat output_frame;
    magnitude(input_frame_1, input_frame_2, output_frame);
    output_channel.write(output_frame);
}

void magnitude_process(WatchChannel<cv::Mat>& input_channel_1, WatchChannel<cv::Mat>& input_channel_2, WatchChannel<cv::Mat>& output_channel, ProcessorState& processorState) {
    DualInputProcessor processor("Magnitude", &processorState);

    processor.register_callback(magnitude_task);
    processor.start(input_channel_1, input_channel_2, output_channel);
}

class MagnitudeTask : public Task {
public:
    explicit MagnitudeTask(WatchChannel<cv::Mat> &outputChannel) : Task(MAGNITUDE, outputChannel) {}

    void start(WatchChannel<cv::Mat>& input_1, WatchChannel<cv::Mat>& input_2) {
        processorThread = std::thread(magnitude_process, std::ref(input_1), std::ref(input_2), std::ref(*outputChannel), std::ref(processorState));
    }
};

#endif //VISION_CPP_MAGNITUDE_H
