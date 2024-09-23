import torch


class DiTi:
    def __init__(self, n_timesteps, K, stages, k_per_stage):
        if k_per_stage:
            k_per_stage = k_per_stage.split(",")
            k_per_stage = [int(k) for k in k_per_stage]
        else:
            k_per_stage = None

        if stages:
            stages = stages.split(",")
            stages = [int(k) for k in stages]
        else:
            stages = None
        self.stages = stages
        self.k_per_stage = k_per_stage

        self.t_to_idx = torch.zeros(n_timesteps).long()
        self.idx_to_max_t = torch.zeros(K).long()
        self.K = K
        if k_per_stage:
            assert stages is not None
            current_stage = 0
            sum_indices = 0
            for t in range(n_timesteps):
                if t == self.stages[current_stage]:
                    sum_indices += self.k_per_stage[current_stage]
                    current_stage += 1
                current_steps = float(self.stages[current_stage])
                current_steps = current_steps - self.stages[current_stage - 1] if current_stage > 0 else current_steps
                current_k = float(self.k_per_stage[current_stage])
                t_adj = t - self.stages[current_stage - 1] if current_stage > 0 else t
                idx = int(float(t_adj) / current_steps * current_k + sum_indices)
                self.t_to_idx[t] = idx
                self.idx_to_max_t[idx] = t
        else:
            for t in range(n_timesteps):
                idx = int(float(t) / (float(n_timesteps) / K))
                self.t_to_idx[t] = idx
                self.idx_to_max_t[idx] = t

    def get_key_timesteps(self):
        return [0] + (self.idx_to_max_t).tolist()

    def get_timestep_range(self, k):
        key_timesteps = self.get_key_timesteps()
        return key_timesteps[k], key_timesteps[k + 1]


if __name__ == "__main__":
    diti = DiTi(1000, 16, "100,600,1000", "2,10,4")
    print(diti.get_key_timesteps())
    print(diti.get_timestep_range(1))

    diti = DiTi(1000, 16, "", "")
    print(diti.get_key_timesteps())
    print(diti.get_timestep_range(1))
