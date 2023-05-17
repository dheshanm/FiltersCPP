// SPDX-FileCopyrightText: 2023 Dheshan Mohandass (L4TTiCe)
//
// SPDX-License-Identifier: MIT

#ifndef VISION_CPP_TASK_H
#define VISION_CPP_TASK_H

#include <string>
#include <opencv2/opencv.hpp>
#include <utility>
#include "../utils/watch_channel.h"
#include "../utils/processor/processor.h"

/**
 * A class that represents a task that can process images from watch channel(s) and send them to another watch channel.
 * It can register a callback function that defines how the images are processed and start the processing loop.
 */
class Task {
public:
    /**
     * A constructor that creates a Task object with a given name and a pointer to a ProcessorState object.
     * @param name
     * @param outputChannel
     */
    Task(std::string name, WatchChannel<cv::Mat> &outputChannel);

    /**
     * A destructor that destroys the Task object.
     */
    virtual ~Task();

    /**
     * A function that can change the running status of the processor.
     * @param value A boolean value that indicates whether the processor should be running or not.
     */
    void set_running(bool value);

    /**
     * A function to display its most recent output frame.
     */
    int display();

    /**
     * A function that returns the state of the processor.
     * @return A ProcessorState object that stores the state of the processor.
     */
    ProcessorState get_state();

    /**
     * A function that returns the output channel of the processor.
     * @return A WatchChannel object that stores the output channel of the processor.
     */
    WatchChannel<cv::Mat> *get_output_channel();

    /**
     * Name of the task.
     */
    std::string name;
protected:
    WatchChannel<cv::Mat> *outputChannel; // output channel of the processor
    ProcessorState processorState; // state of the processor
    std::thread processorThread; // thread that runs the processing loop
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

WatchChannel<cv::Mat> *Task::get_output_channel() {
    return outputChannel;
}

Task::Task(std::string name, WatchChannel<cv::Mat> &outputChannel) {
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
    std::cout << "Destroying task " << name << std::endl;
    cv::destroyWindow(name);
    processorState.running = false;
}

#endif //VISION_CPP_TASK_H
