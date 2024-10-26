#include "CoreAgentLayer.h"

CoreAgentLayer core_agent_layer(NUM_CORES);

void prefetcher_on_cache_access(int core_id, uint64_t address, bool cache_hit) {
    std::vector<float> state = extract_state_from_address(address);
    core_agent_layer.update_state(core_id, state);
    core_agent_layer.compute_reward(core_id, cache_hit);
    int prefetch_distance = core_agent_layer.agents[core_id].policy_network->predict(state);
    prefetch_request(core_id, address + prefetch_distance);
    core_agent_layer.update_policy(core_id);
}

std::vector<float> extract_state_from_address(uint64_t address) {
    std::vector<float> state(10, 0.0f);
    for (int i = 0; i < 10; ++i) {
        state[i] = static_cast<float>((address >> (i * 5)) & 0x1F);
    }
    return state;
}
