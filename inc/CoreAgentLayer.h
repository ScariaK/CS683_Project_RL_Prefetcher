#pragma once
#include <vector>
#include <unordered_map>
#include <cmath>
#include "LLPN.h" 

class CoreAgentLayer {
public:
    struct Agent {
        int core_id;
        LLPN* policy_network; 
        std::vector<float> state; 
        float reward;
        float discount_factor;
        Agent(int id) : core_id(id), policy_network(new LLPN()), reward(0.0f), discount_factor(0.9f) {}
        ~Agent() { delete policy_network; }
    };
    std::vector<Agent> agents;
    CoreAgentLayer(int num_cores) {
        for (int i = 0; i < num_cores; ++i) {
            agents.push_back(Agent(i));
        }
    }
    void update_state(int core_id, std::vector<float> new_state) {
        agents[core_id].state = std::move(new_state);
    }
    void compute_reward(int core_id, bool cache_hit) {
        agents[core_id].reward = cache_hit ? 1.0f : -1.0f;
    }
    void update_policy(int core_id) {
        agents[core_id].policy_network->train(agents[core_id].state, agents[core_id].reward);
    }
};
