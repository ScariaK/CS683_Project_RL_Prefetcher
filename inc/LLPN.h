#pragma once
#include <vector>

class LLPN {
public:
    std::vector<float> weights;
    float learning_rate;

    LLPN() : learning_rate(0.01f) {
        weights.resize(10, 0.1f);
    }
    //predictig the next prefetching distance
    int predict(const std::vector<float>& state) {
        float sum = 0.0f;
        for (int i = 0; i < state.size(); ++i) {
            sum += state[i] * weights[i];
        }
        return static_cast<int>(std::round(sum));
    }
     // Training function
    void train(const std::vector<float>& state, float reward) {
        for (int i = 0; i < state.size(); ++i) {
            weights[i] += learning_rate * reward * state[i];
        }
    }
};
