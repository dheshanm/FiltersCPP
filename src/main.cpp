// SPDX-FileCopyrightText: 2023 Dheshan Mohandass (L4TTiCe)
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <unordered_map>

#include "constants.h"
#include "utils/camera/camera.h"
#include "tasks/greyscale.h"
#include "tasks/negative.h"
#include "tasks/blur.h"
#include "tasks/sobel.h"
#include "tasks/magnitude.h"
#include "tasks/quantize.h"
#include "tasks/cartoonize.h"

[[noreturn]] void fetch_frame(Camera &camera, WatchChannel<cv::Mat> &watchChannel) {
    while (true) {
        cv::Mat frame;
        camera.read(frame);
        if (frame.empty()) {
            std::cout << "Fetch: " << "Failed to capture frame." << std::endl;
            continue;
        }
        watchChannel.write(frame);
    }
}

int display_channel(WatchChannel<cv::Mat> &watchChannel, const std::string &window_name) {
    cv::Mat frame;
    watchChannel.read(frame);

    if (frame.empty()) {
        return -1;
    }
    cv::imshow(window_name, frame);

    return 0;
}

void
create_channel(std::unordered_map<std::string, WatchChannel<cv::Mat> *> &channels, const std::string &channel_name) {
    if (channels.find(channel_name) != channels.end()) {
        return;
    }
    channels[channel_name] = new WatchChannel<cv::Mat>();
}

void stop_task(const std::string &task_name, std::unordered_map<std::string, Task *> &tasks) {
    tasks[task_name]->set_running(false);
    tasks.erase(task_name);
    cv::destroyWindow(task_name);

    std::cout << "Stopped " << task_name << std::endl;
}

void start_sobel_tasks(std::unordered_map<std::string, Task *> &tasks,
                       std::unordered_map<std::string, WatchChannel<cv::Mat> *> &channels) {
    create_channel(channels, SOBEL_X);
    auto *sobelXTask = new SobelXTask(*channels[SOBEL_X]);
    tasks[SOBEL_X] = sobelXTask;
    sobelXTask->start(*channels[MAIN]);

    std::cout << "Started Sobel X" << std::endl;

    create_channel(channels, SOBEL_Y);
    auto *sobelYTask = new SobelYTask(*channels[SOBEL_Y]);
    tasks[SOBEL_Y] = sobelYTask;
    sobelYTask->start(*channels[MAIN]);

    std::cout << "Started Sobel Y" << std::endl;

    create_channel(channels, MAGNITUDE);
    auto *magnitudeTask = new MagnitudeTask(*channels[MAGNITUDE]);
    tasks[MAGNITUDE] = magnitudeTask;
    magnitudeTask->start(*channels[SOBEL_X], *channels[SOBEL_Y]);

    std::cout << "Started Magnitude" << std::endl;
}

void stop_sobel_tasks(std::unordered_map<std::string, Task *> &tasks) {
    stop_task(SOBEL_X, tasks);
    stop_task(SOBEL_Y, tasks);
    stop_task(MAGNITUDE, tasks);
}

void start_quantize_task(std::unordered_map<std::string, Task *> &tasks,
                         std::unordered_map<std::string, WatchChannel<cv::Mat> *> &channels) {
    channels[QUANTIZED] = new WatchChannel<cv::Mat>();
    auto *quantizedTask = new QuantizedTask(*channels[QUANTIZED]);
    tasks[QUANTIZED] = quantizedTask;
    quantizedTask->start(*channels[MAIN]);

    std::cout << "Started Quantized" << std::endl;
}

int main() {
    Camera camera(0);
    camera.set_fps(30);

    std::unordered_map<std::string, WatchChannel<cv::Mat> *> channels;
    std::unordered_map<std::string, Task *> tasks;

    channels[MAIN] = new WatchChannel<cv::Mat>();

    int key_pressed;

    std::thread fetch_thread(fetch_frame, std::ref(camera), std::ref(*channels[MAIN]));

    bool is_running = true;
    while (is_running) {
        display_channel(*channels[MAIN], MAIN);

        for (auto &pair: tasks) {
            pair.second->display();
        }

        key_pressed = cv::waitKey(1);
        switch (key_pressed) {
            case -1: {
                break;
            }
            case 98: { // b
                std::cout << "Key pressed: [B] " << key_pressed << std::endl;

                if (tasks.find(BLUR) == tasks.end()) {
                    create_channel(channels, BLUR);
                    auto *blurTask = new BlurTask(*channels[BLUR]);
                    tasks[BLUR] = blurTask;
                    blurTask->start(*channels[MAIN]);

                    std::cout << "Started Blur" << std::endl;
                } else {
                    tasks.erase(BLUR);
                    channels.erase(BLUR);
                    cv::destroyWindow(BLUR);

                    std::cout << "Stopped Blur" << std::endl;
                }

                break;
            }
            case 99: { // c
                std::cout << "Key pressed: [C] " << key_pressed << std::endl;

                if (tasks.find(CARTOONIZE) != tasks.end()) {
                    stop_task(CARTOONIZE, tasks);
                } else {
                    if (tasks.find(MAGNITUDE) == tasks.end()) {
                        start_sobel_tasks(tasks, channels);
                    }
                    if (tasks.find(QUANTIZED) == tasks.end()) {
                        start_quantize_task(tasks, channels);
                    }

                    create_channel(channels, CARTOONIZE);
                    auto *cartoonizeTask = new CartoonizeTask(*channels[CARTOONIZE]);
                    tasks[CARTOONIZE] = cartoonizeTask;
                    cartoonizeTask->start(*channels[QUANTIZED], *channels[MAGNITUDE]);

                    std::cout << "Started Cartoonize" << std::endl;
                }

                break;
            }
            case 102: { // f
                std::cout << "Key pressed: [F] " << key_pressed << std::endl;
                for (auto &pair: tasks) {
                    std::cout << pair.first << ": " << pair.second->get_state().fps_counter << " fps ("
                              << pair.second->get_state().frame_time << "ms )" << std::endl;
                }
                break;
            }
            case 103: { // g
                std::cout << "Key pressed: [G] " << key_pressed << std::endl;

                if (tasks.find(GRAYSCALE) == tasks.end()) {
                    create_channel(channels, GRAYSCALE);
                    auto *grayscaleTask = new GrayscaleTask(*channels[GRAYSCALE]);
                    tasks[GRAYSCALE] = grayscaleTask;
                    grayscaleTask->start(*channels[MAIN]);

                    std::cout << "Started Grayscale" << std::endl;
                } else {
                    stop_task(GRAYSCALE, tasks);
                }

                break;
            }
            case 110: { // n
                std::cout << "Key pressed: [N] " << key_pressed << std::endl;

                if (tasks.find(NEGATIVE) == tasks.end()) {
                    create_channel(channels, NEGATIVE);
                    auto *negativeTask = new NegativeTask(*channels[NEGATIVE]);
                    tasks[NEGATIVE] = negativeTask;
                    negativeTask->start(*channels[MAIN]);

                    std::cout << "Started Negative" << std::endl;
                } else {
                    stop_task(NEGATIVE, tasks);
                }

                break;
            }
            case 113: { // q
                std::cout << "Key pressed: [Q] " << key_pressed << std::endl;

                if (tasks.find(QUANTIZED) == tasks.end()) {
                    start_quantize_task(tasks, channels);
                } else {
                    stop_task(QUANTIZED, tasks);
                }


                break;
            }
            case 115: { // s
                std::cout << "Key pressed: [S] " << key_pressed << std::endl;

                if (tasks.find(SOBEL_X) == tasks.end()) {
                    start_sobel_tasks(tasks, channels);
                } else {
                    stop_sobel_tasks(tasks);
                }

                break;
            }
            default:
                std::cout << "Key pressed: " << key_pressed << std::endl;
                std::cout << "Exiting..." << std::endl;
                is_running = false;
                cv::destroyAllWindows();
                break;
        }
    }

    return 0;
}
