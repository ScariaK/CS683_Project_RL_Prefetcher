#include <iostream>
#include <queue>
#include <unordered_map>
#include <vector>
#include <cassert>

struct QueueEntry {
    uint32_t prefetched_address;
    bool cache_hit;
    float reward;
    uint32_t state;
    uint32_t offset;
};

class RewardFIFOQueue {
public:
    RewardFIFOQueue(size_t capacity, TRTable &reward_table)
        : m_capacity(capacity), m_reward_table(reward_table) {}

    void issuePrefetch(uint32_t prefetched_address, uint32_t state, uint32_t offset) {
        bool cache_hit = false;
        float reward = 0.0f;

        
        auto it = m_address_map.find(prefetched_address);
        if (it != m_address_map.end()) {
            QueueEntry &entry = m_queue[it->second];
            if (entry.cache_hit) {
                // Update reward based on cache hit
                reward = accurate_prefetch_reward;
            } else {
                reward = no_prefetch_reward;
            }
        } else {
            if (m_queue.size() >= m_capacity) {
                evictEntry(); 
            }
            m_queue.push({prefetched_address, cache_hit, reward, state, offset});
            m_address_map[prefetched_address] = m_queue.size() - 1;
        }
    }

    void processCacheHit(uint32_t address) {
        auto it = m_address_map.find(address);
        if (it != m_address_map.end()) {
            QueueEntry &entry = m_queue[it->second];
            entry.cache_hit = true;
            entry.reward = accurate_prefetch_reward;
        }
    }

private:
    void evictEntry() {
        if (m_queue.empty()) return;

        QueueEntry entry = m_queue.front();
        m_queue.pop();
        m_address_map.erase(entry.prefetched_address);

        // Use state-offset pair to update total reward in the table
        float updated_reward = entry.reward;
        m_reward_table.learn(entry.state, entry.offset, updated_reward, entry.state, entry.offset);
    }

    size_t m_capacity;
    TRTable &m_reward_table;
    std::queue<QueueEntry> m_queue;
    std::unordered_map<uint32_t, size_t> m_address_map;

    const float accurate_prefetch_reward = 1.0f;
    const float no_prefetch_reward = 0.0f;
};
