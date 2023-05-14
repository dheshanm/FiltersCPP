//
// Created by master on 5/14/23.
//

#ifndef VISION_CPP_WATCH_CHANNEL_H
#define VISION_CPP_WATCH_CHANNEL_H

#include <mutex>

template <typename T>
class WatchChannel {
public:
    explicit WatchChannel();
    ~WatchChannel();
    int read(T& output);
    int write(T& input);

private:
    T data;
    std::mutex mutex;
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
