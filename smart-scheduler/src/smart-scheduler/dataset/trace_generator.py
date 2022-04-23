import numpy as np
import random
from tqdm import tqdm
from typing import Dict, Any

# from smart_scheduler.network import (
#     NetworkSimulatorRandom,
#     NetworkSimulatorDataset
# )
from smart_scheduler.network import (
    NetworkBuilderRandom,
    NetworkBuilderDataset
)


class TraceGenerator:
    def __init__(self, edge_simulator_config: Dict[str, Any],
                 timesteps: int,
                 from_dataset: bool, seed: int):
        """
            dataset generator
        """
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(seed)

        if from_dataset:
            self.sim = NetworkBuilderDataset.with_network(**edge_simulator_config)
        else:
            self.sim = NetworkBuilderRandom.with_network(**edge_simulator_config)
        self.timesteps = timesteps

    def make_traces(self):
        """
        make the movement traces
        """
        traces = []
        for col in tqdm(range(0, self.timesteps)):
            self.sim.sample_users_stations(from_trace=False)
            traces.append(self.sim.users_location)
        return traces
