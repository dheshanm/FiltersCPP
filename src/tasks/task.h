//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_TASK_H
#define VISION_CPP_TASK_H

#include <string>
#include <opencv2/opencv.hpp>
#include <utility>
#include "../utils/watch_channel.h"
#include "../utils/processor/processor.h"

class Task {
public:
    Task(std::string name, WatchChannel<cv::Mat>& outputChannel);
    virtual ~Task();
    void set_running(bool value);
    int display();
    ProcessorState get_state();
    WatchChannel<cv::Mat>* get_output_channel();
    std::string name;
protected:
    WatchChannel<cv::Mat>* outputChannel;
    ProcessorState processorState;
    std::thread processorThread;
};

int Task::display() {
    cv::Mat frame;
    outputChannel->read(frame);

    if (frame.empty()) {
        return -1;
    }
    cv::imshow(name, frame);

    return 0;
}

WatchChannel<cv::Mat>* Task::get_output_channel() {
    return outputChannel;
}

Task::Task(std::string name, WatchChannel<cv::Mat>& outputChannel) {
    this->name = std::move(name);
    this->outputChannel = &outputChannel;
}

void Task::set_running(bool value) {
    processorState.running = value;
}

ProcessorState Task::get_state() {
    return processorState;
}


Task::~Task() {
    cv::destroyWindow(name);
}

#endif //VISION_CPP_TASK_H
