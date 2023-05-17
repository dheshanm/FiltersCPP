// SPDX-FileCopyrightText: 2023 Dheshan Mohandass (L4TTiCe)
//
// SPDX-License-Identifier: MIT

#ifndef VISION_CPP_PROCESSOR_H
#define VISION_CPP_PROCESSOR_H

#include <string>
#include <opencv2/core/mat.hpp>
#include "../watch_channel.h"

/**
 * A class that represents the state of a processor.
 * It contains information about the running status, the frames per second and the frame time of the processor.
 */
class ProcessorState {
public:
    /**
     * A constructor that creates a ProcessorState object with default values.
     */
    explicit ProcessorState();

    /**
     * A destructor that destroys the ProcessorState object.
     */
    ~ProcessorState();

    /**
     * A boolean variable that indicates whether the processor is running or not.
     */
    bool running;

    /**
     * An integer variable that counts the number of frames processed per second by the processor.
     */
    int fps_counter;

    /**
     * An integer variable that measures the time taken to process one frame by the processor in milliseconds.
     */
    int frame_time;
};


/**
 * A class that represents a processor that can process images from a watch channel and send them to another watch channel.
 * It can register a callback function that defines how the images are processed and start the processing loop.
 */
class Processor {
public:
    /**
     * A constructor that creates a Processor object with a given name and a pointer to a ProcessorState object.
     * @param name A string that specifies the name of the processor.
     * @param state A pointer to a ProcessorState object that stores the state of the processor.
     */
    explicit Processor(std::string name, ProcessorState *state);

    /**
     * A destructor that destroys the Processor object and frees the memory allocated for the ProcessorState object.
     */
    ~Processor();

    /**
     * A method that registers a callback function that defines how the images are processed by the processor.
     * The callback function takes two parameters: a reference to a WatchChannel<cv::Mat> object that provides the input images,
     * and a reference to another WatchChannel<cv::Mat> object that receives the output images.
     * @param callback A pointer to a function that takes two parameters: a reference to a WatchChannel<cv::Mat> object and another reference to a WatchChannel<cv::Mat> object.
     * @return An integer value that indicates whether the registration was successful or not. Zero means success, non-zero means failure.
     */
    int register_callback(void (*callback)(WatchChannel<cv::Mat> &input, WatchChannel<cv::Mat> &output));

    /**
     * A method that starts the processing loop of the processor.
     * It reads images from the input watch channel, passes them to the callback function, and writes the results to the output watch channel.
     * It also updates the state of the processor according to the running status, the frames per second and the frame time.
     * @param input A reference to a WatchChannel<cv::Mat> object that provides the input images for the processor.
     * @param output A reference to another WatchChannel<cv::Mat> object that receives the output images from the processor.
     * @return An integer value that indicates whether the processing loop was started successfully or not. Zero means success, non-zero means failure.
     */
    int start(WatchChannel<cv::Mat> &input, WatchChannel<cv::Mat> &output);

private:
    /**
     * A string variable that stores the name of the processor.
     */
    std::string name;

    /**
     * A pointer to a ProcessorState object that stores the state of the processor.
     */
    ProcessorState *state;

    /**
     * A pointer to a function that defines how the images are processed by the processor.
     */
    void (*callback)(WatchChannel<cv::Mat> &input, WatchChannel<cv::Mat> &output);
};

/**
 * A class that represents a processor that can process images from two watch channels and send them to another watch channel.
 * It can register a callback function that defines how the images are processed and start the processing loop.
 */
class DualInputProcessor {
public:
    /**
     * A constructor that creates a DualInputProcessor object with a given name and a pointer to a ProcessorState object.
     * @param name A string that specifies the name of the processor.
     * @param state A pointer to a ProcessorState object that stores the state of the processor.
     */
    explicit DualInputProcessor(std::string name, ProcessorState *state);

    /**
     * A destructor that destroys the DualInputProcessor object and frees the memory allocated for the ProcessorState object.
     */
    ~DualInputProcessor();

    /**
     * A method that registers a callback function that defines how the images are processed by the processor.
     * The callback function takes three parameters: two references to WatchChannel<cv::Mat> objects that provide the input images,
     * and a reference to another WatchChannel<cv::Mat> object that receives the output images.
     * @param callback A pointer to a function that takes three parameters: two references to WatchChannel<cv::Mat> objects and another reference to a WatchChannel<cv::Mat> object.
     * @return An integer value that indicates whether the registration was successful or not. Zero means success, non-zero means failure.
     */
    int register_callback(void (*callback)(WatchChannel<cv::Mat> &input_1, WatchChannel<cv::Mat> &input_2,
                                           WatchChannel<cv::Mat> &output));

    /**
     * A method that starts the processing loop of the processor.
     * It reads images from the two input watch channels, passes them to the callback function, and writes the results to the output watch channel.
     * It also updates the state of the processor according to the running status, the frames per second and the frame time.
     * @param input_1 A reference to a WatchChannel<cv::Mat> object that provides the first input images for the processor.
     * @param input_2 A reference to another WatchChannel<cv::Mat> object that provides the second input images for the processor.
     * @param output A reference to another WatchChannel<cv::Mat> object that receives the output images from the processor.
     * @return An integer value that indicates whether the processing loop was started successfully or not. Zero means success, non-zero means failure.
     */
    int start(WatchChannel<cv::Mat> &input_1, WatchChannel<cv::Mat> &input_2, WatchChannel<cv::Mat> &output);

private:
    /**
     * A string variable that stores the name of the processor.
     */
    std::string name;

    /**
     * A pointer to a ProcessorState object that stores the state of the processor.
     */
    ProcessorState *state;

    /**
     * A pointer to a function that defines how the images are processed by the processor.
     */
    void (*callback)(WatchChannel<cv::Mat> &input_1, WatchChannel<cv::Mat> &input_2, WatchChannel<cv::Mat> &output);
};


#endif //VISION_CPP_PROCESSOR_H
