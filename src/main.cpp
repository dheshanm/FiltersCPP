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

[[noreturn]] void fetch_frame(Camera& camera, WatchChannel<cv::Mat>& watchChannel) {
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

int display_channel(WatchChannel<cv::Mat>& watchChannel, const std::string& window_name) {
    cv::Mat frame;
    watchChannel.read(frame);

    if (frame.empty()) {
        return -1;
    }
    cv::imshow(window_name, frame);

    return 0;
}

int main() {
    Camera camera(0);
    camera.set_fps(30);

    std::unordered_map<std::string, WatchChannel<cv::Mat>*> channels;
    std::unordered_map<std::string, Task*> tasks;

    channels[MAIN] = new WatchChannel<cv::Mat>();

    int key_pressed;

    std::thread fetch_thread(fetch_frame, std::ref(camera), std::ref(*channels[MAIN]));

    bool is_running = true;
    while (is_running) {
        display_channel(*channels[MAIN], MAIN);

        for (auto& pair : tasks) {
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
                    channels[BLUR] = new WatchChannel<cv::Mat>();
                    auto* blurTask = new BlurTask(*channels[BLUR]);
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
            case 102: { // f
                std::cout << "Key pressed: [F] " << key_pressed << std::endl;
                for (auto& pair : tasks) {
                    std::cout<< pair.first << ": " << pair.second->get_state().fps_counter << " fps (" << pair.second->get_state().frame_time << "ms )" << std::endl;
                }
                break;
            }
            case 103: { // g
                std::cout << "Key pressed: [G] " << key_pressed << std::endl;

                if (tasks.find(GRAYSCALE) == tasks.end()) {
                    channels[GRAYSCALE] = new WatchChannel<cv::Mat>();
                    auto* grayscaleTask = new GrayscaleTask(*channels[GRAYSCALE]);
                    tasks[GRAYSCALE] = grayscaleTask;
                    grayscaleTask->start(*channels[MAIN]);

                    std::cout << "Started Grayscale" << std::endl;
                } else {
                    tasks.erase(GRAYSCALE);
                    channels.erase(GRAYSCALE);
                    cv::destroyWindow(GRAYSCALE);

                    std::cout << "Stopped Grayscale" << std::endl;
                }

                break;
            }
            case 110: { // n
                std::cout << "Key pressed: [N] " << key_pressed << std::endl;

                if (tasks.find(NEGATIVE) == tasks.end()) {
                    channels[NEGATIVE] = new WatchChannel<cv::Mat>();
                    auto* negativeTask = new NegativeTask(*channels[NEGATIVE]);
                    tasks[NEGATIVE] = negativeTask;
                    negativeTask->start(*channels[MAIN]);

                    std::cout << "Started Negative" << std::endl;
                } else {
                    tasks.erase(NEGATIVE);
                    channels.erase(NEGATIVE);
                    cv::destroyWindow(NEGATIVE);

                    std::cout << "Stopped Negative" << std::endl;
                }

                break;
            }
            case 115: { // s
                std::cout << "Key pressed: [S] " << key_pressed << std::endl;

                if (tasks.find(SOBEL_X) == tasks.end()) {
                    channels[SOBEL_X] = new WatchChannel<cv::Mat>();
                    auto* sobelXTask = new SobelXTask(*channels[SOBEL_X]);
                    tasks[SOBEL_X] = sobelXTask;
                    sobelXTask->start(*channels[MAIN]);

                    std::cout << "Started Sobel X" << std::endl;
                } else {
                    tasks.erase(SOBEL_X);
                    channels.erase(SOBEL_X);
                    cv::destroyWindow(SOBEL_X);

                    std::cout << "Stopped Sobel X" << std::endl;
                }

                if (tasks.find(SOBEL_Y) == tasks.end()) {
                    channels[SOBEL_Y] = new WatchChannel<cv::Mat>();
                    auto* sobelYTask = new SobelYTask(*channels[SOBEL_Y]);
                    tasks[SOBEL_Y] = sobelYTask;
                    sobelYTask->start(*channels[MAIN]);

                    std::cout << "Started Sobel Y" << std::endl;
                } else {
                    tasks.erase(SOBEL_Y);
                    channels.erase(SOBEL_Y);
                    cv::destroyWindow(SOBEL_Y);

                    std::cout << "Stopped Sobel Y" << std::endl;
                }

                if (tasks.find(MAGNITUDE) == tasks.end()) {
                    channels[MAGNITUDE] = new WatchChannel<cv::Mat>();
                    auto* magnitudeTask = new MagnitudeTask(*channels[MAGNITUDE]);
                    tasks[MAGNITUDE] = magnitudeTask;
                    magnitudeTask->start(*channels[SOBEL_X], *channels[SOBEL_Y]);

                    std::cout << "Started Magnitude" << std::endl;
                } else {
                    tasks.erase(MAGNITUDE);
                    channels.erase(MAGNITUDE);
                    cv::destroyWindow(MAGNITUDE);

                    std::cout << "Stopped Magnitude" << std::endl;
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
