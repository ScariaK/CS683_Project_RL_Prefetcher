#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <sstream>
#include <vector>
#include <random>

class TRTable {
public:
    TRTable(uint32_t offsets, uint32_t states, float gamma, uint64_t seed, std::string policy, bool zero_init, uint64_t early_exploration_window)
        : m_offsets(offsets), m_states(states), m_gamma(gamma), m_policy(policy), m_early_exploration_window(early_exploration_window), m_offset_counter(0)
    {
        reward_table = (float**)calloc(m_states, sizeof(float*));
        assert(reward_table);
        for (uint32_t index = 0; index < m_states; ++index) {
            reward_table[index] = (float*)calloc(m_offsets, sizeof(float));
            assert(reward_table[index]);
        }

        // Initialize reward table
        if (zero_init) {
            init_value = 0;
        } else {
            init_value = (float)1ul / (1 - m_gamma);
        }

        for (uint32_t row = 0; row < m_states; ++row) {
            for (uint32_t col = 0; col < m_offsets; ++col) {
                reward_table[row][col] = init_value;
            }
        }

        generator.seed(seed);
        explore = new std::bernoulli_distribution(0.1f);  // example epsilon for exploration
        offsetgen = new std::uniform_int_distribution<int>(0, m_offsets - 1);
    }

    ~TRTable() {
        for (uint32_t row = 0; row < m_states; ++row) {
            free(reward_table[row]);
        }
        free(reward_table);
    }

    uint32_t chooseOffset(uint32_t state) {
        assert(state < m_states);
        uint32_t offset = 0;
        if (m_policy == "e-greedy") {
            if (m_offset_counter < m_early_exploration_window || (*explore)(generator)) {
                offset = (*offsetgen)(generator);  // random offset
                m_offset_counter++;
            } else {
                offset = getMaxOffset(state);  // best offset based on reward table
            }
        } else {
           
            assert(false);
        }

        return offset;
    }

    void learn(uint32_t state1, uint32_t offset1, int32_t reward, uint32_t state2, uint32_t offset2) {
        float Rsa1, Rsa2, Rsa1_old;
        Rsa1 = consultRewardTable(state1, offset1);
        Rsa2 = consultRewardTable(state2, offset2);
        Rsa1_old = Rsa1;
        Rsa1 = Rsa1 + m_gamma * ((float)reward + m_gamma * Rsa2 - Rsa1);
        updateRewardTable(state1, offset1, Rsa1);
    }

private:
    uint32_t m_offsets;
    uint32_t m_states;
    float m_gamma;
    float init_value;
    uint64_t m_offset_counter;
    uint64_t m_early_exploration_window;
    std::string m_policy;

    float** reward_table;
    std::default_random_engine generator;
    std::bernoulli_distribution* explore;
    std::uniform_int_distribution<int>* offsetgen;

    float consultRewardTable(uint32_t state, uint32_t offset) {
        assert(state < m_states && offset < m_offsets);
        return reward_table[state][offset];
    }

    void updateRewardTable(uint32_t state, uint32_t offset, float value) {
        assert(state < m_states && offset < m_offsets);
        reward_table[state][offset] = value;
    }

    uint32_t getMaxOffset(uint32_t state) {
        assert(state < m_states);
        float max = reward_table[state][0];
        uint32_t offset = 0;
        for (uint32_t index = 1; index < m_offsets; ++index) {
            if (reward_table[state][index] > max) {
                max = reward_table[state][index];
                offset = index;
            }
        }
        return offset;
    }
};
