//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_PROCESSOR_H
#define VISION_CPP_PROCESSOR_H

#include <string>
#include <opencv2/core/mat.hpp>
#include "../watch_channel.h"

class ProcessorState {
public:
    explicit ProcessorState();
    ~ProcessorState();
    bool running;
    int fps_counter;
    int frame_time;
};

class Processor {
public:
    explicit Processor(std::string name, ProcessorState* state);
    ~Processor();
    int register_callback(void (*callback)(WatchChannel<cv::Mat> &input, WatchChannel<cv::Mat> &output));

    int start(WatchChannel<cv::Mat> &input, WatchChannel<cv::Mat> &output);

private:
    std::string name;
    ProcessorState* state;
    void (*callback)(WatchChannel<cv::Mat> &input, WatchChannel<cv::Mat> &output);
};

class DualInputProcessor {
public:
    explicit DualInputProcessor(std::string name, ProcessorState* state);
    ~DualInputProcessor();
    int register_callback(void (*callback)(WatchChannel<cv::Mat> &input_1, WatchChannel<cv::Mat> &input_2, WatchChannel<cv::Mat> &output));

    int start(WatchChannel<cv::Mat> &input_1, WatchChannel<cv::Mat> &input_2, WatchChannel<cv::Mat> &output);

private:
    std::string name;
    ProcessorState* state;
    void (*callback)(WatchChannel<cv::Mat> &input_1, WatchChannel<cv::Mat> &input_2, WatchChannel<cv::Mat> &output);
};


#endif //VISION_CPP_PROCESSOR_H
