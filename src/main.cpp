#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

#include "constants.h"
#include "utils/camera/camera.h"
#include "tasks/greyscale.h"
#include "tasks/negative.h"
#include "tasks/blur.h"

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

    WatchChannel<cv::Mat> watchChannel;
    WatchChannel<cv::Mat> grayscaleChannel;
    WatchChannel<cv::Mat> negativeChannel;
    WatchChannel<cv::Mat> blurChannel;
    int key_pressed;

    ProcessorState grayscaleProcessorState;
    ProcessorState negativeProcessorState;
    ProcessorState blurProcessorState;

    std::thread fetch_thread(fetch_frame, std::ref(camera), std::ref(watchChannel));
    std::thread grayscale_thread(grayscale_process, std::ref(watchChannel), std::ref(grayscaleChannel), std::ref(grayscaleProcessorState));
    std::thread negative_thread(negative_process, std::ref(watchChannel), std::ref(negativeChannel), std::ref(negativeProcessorState));
    std::thread blur_thread(blur_process, std::ref(watchChannel), std::ref(blurChannel), std::ref(blurProcessorState));

    bool is_running = true;
    while (is_running) {
        display_channel(watchChannel, WINDOW_NAME);
        display_channel(grayscaleChannel, WINDOW_NAME_GRAYSCALE);
        display_channel(negativeChannel, WINDOW_NAME_NEGATIVE);
        display_channel(blurChannel, WINDOW_NAME_BLUR);

        key_pressed = cv::waitKey(1);
        switch (key_pressed) {
            case -1: {
                break;
            }
            case 98: { // b
                std::cout << "Key pressed: [B] " << key_pressed << std::endl;
                blurProcessorState.running = !blurProcessorState.running;
                std::cout << "Toggling Blur: running:" << blurProcessorState.running << std::endl;
                break;
            }
            case 102: { // f
                std::cout << "Key pressed: [F] " << key_pressed << std::endl;
                std::cout << "GrayScale: " << grayscaleProcessorState.fps_counter << " fps (" << grayscaleProcessorState.frame_time << "ms )" << std::endl;
                std::cout << "Negative: " << negativeProcessorState.fps_counter << " fps (" << negativeProcessorState.frame_time << "ms )" << std::endl;
                std::cout << "Blur: " << blurProcessorState.fps_counter << " fps (" << blurProcessorState.frame_time << "ms )" << std::endl;
                break;
            }
            case 103: { // g
                std::cout << "Key pressed: [G] " << key_pressed << std::endl;
                grayscaleProcessorState.running = !grayscaleProcessorState.running;
                std::cout << "Toggling Grayscale: running:" << grayscaleProcessorState.running << std::endl;
                break;
            }
            case 110: { // n
                std::cout << "Key pressed: [N] " << key_pressed << std::endl;
                negativeProcessorState.running = !negativeProcessorState.running;
                std::cout << "Toggling Negative: running:" << negativeProcessorState.running << std::endl;
                break;
            }
            default:
                std::cout << "Key pressed: " << key_pressed << std::endl;
                std::cout << "Exiting..." << std::endl;
                is_running = false;
                break;
        }
    }

    return 0;
}
