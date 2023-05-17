// SPDX-FileCopyrightText: 2023 Dheshan Mohandass (L4TTiCe)
//
// SPDX-License-Identifier: MIT

#ifndef VISION_CPP_WATCH_CHANNEL_H
#define VISION_CPP_WATCH_CHANNEL_H

#include <mutex>

/**
 * A template class that implements a thread-safe channel for data exchange.
 * A channel is a one-way communication mechanism that allows one thread to send data to another thread.
 * The channel has a buffer of size one, which means it can store only one data item at a time.
 * The channel supports read and write operations, which are synchronized using a mutex.
 * @tparam T The type of data that the channel can hold.
 */
template<typename T>
class WatchChannel {
public:
    /**
    * The default constructor of the channel.
    */
    explicit WatchChannel();

    /**
    * The destructor of the channel.
    */
    ~WatchChannel();

    /**
    * Reads the data from the channel and stores it in the output parameter.
    * This operation blocks until the channel has some data to read.
    * @param output A reference to a variable of type T where the data will be stored.
    * @return 0 if the read operation is successful, or a non-zero error code otherwise.
    */
    int read(T &output);

    /**
    * Writes the data to the channel from the input parameter.
    * This operation blocks until the channel has some space to write.
    * @param input A reference to a variable of type T that contains the data to be written.
    * @return 0 if the write operation is successful, or a non-zero error code otherwise.
    */
    int write(T &input);

private:
    T data; // The buffer that holds the data
    std::mutex mutex; // The mutex that synchronizes the read and write operations
};

template<typename T>
WatchChannel<T>::WatchChannel() = default;

template<typename T>
WatchChannel<T>::~WatchChannel() = default;

template<typename T>
int WatchChannel<T>::read(T &output) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    output = this->data;
    return 0;
}

template<typename T>
int WatchChannel<T>::write(T &input) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    this->data = input;
    return 0;
}

#endif //VISION_CPP_WATCH_CHANNEL_H
