//
// Created by master on 5/14/23.
//

#include "processor.h"

#include <iostream>
#include <chrono>

ProcessorState::ProcessorState() {
    this->running = true;
    this->fps_counter = 0;
    this->frame_time = 0;
}

ProcessorState::~ProcessorState() = default;

Processor::Processor(std::string name, ProcessorState *state) {
    this->name = std::move(name);
    this->state = state;
    this->callback = nullptr;
}

int Processor::register_callback(void (*callback_input)(WatchChannel<cv::Mat> &input, WatchChannel<cv::Mat> &output)) {
    this->callback = callback_input;
    return 0;
}

int Processor::start(WatchChannel<cv::Mat> &input, WatchChannel<cv::Mat> &output) {

    if (this->callback == nullptr) {
        std::cout << "Callback not registered." << std::endl;
        return -1;
    }
    this->state->fps_counter = 0;
    this->state->frame_time = 0;

    int frames_counter = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (true) {
        if (!this->state->running) {
            continue;
        }
        auto frame_time_start = std::chrono::high_resolution_clock::now();
        frames_counter++;

        this->callback(input, output);

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (time >= 1000) {
            this->state->fps_counter = frames_counter;
            frames_counter = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        this->state->frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - frame_time_start).count();
    }
}

Processor::~Processor() = default;

DualInputProcessor::DualInputProcessor(std::string name, ProcessorState *state) {
    this->name = std::move(name);
    this->state = state;
    this->callback = nullptr;
}

int DualInputProcessor::register_callback(void (*callback_input)(WatchChannel<cv::Mat> &input_1, WatchChannel<cv::Mat> &input_2, WatchChannel<cv::Mat> &output)) {
    this->callback = callback_input;
    return 0;
}

int DualInputProcessor::start(WatchChannel<cv::Mat> &input_1, WatchChannel<cv::Mat> &input_2, WatchChannel<cv::Mat> &output) {

    if (this->callback == nullptr) {
        std::cout << "Callback not registered." << std::endl;
        return -1;
    }
    this->state->fps_counter = 0;
    this->state->frame_time = 0;

    int frames_counter = 0;
    auto start = std::chrono::high_resolution_clock::now();

    while (true) {
        if (!this->state->running) {
            continue;
        }
        auto frame_time_start = std::chrono::high_resolution_clock::now();
        frames_counter++;

        this->callback(input_1, input_2, output);

        auto end = std::chrono::high_resolution_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        if (time >= 1000) {
            this->state->fps_counter = frames_counter;
            frames_counter = 0;
            start = std::chrono::high_resolution_clock::now();
        }

        this->state->frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - frame_time_start).count();
    }
}

DualInputProcessor::~DualInputProcessor() = default;
