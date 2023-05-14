import torch
import numpy as np
from naslib.optimizers import RegularizedEvolution


class RegularizedEvolution(RegularizedEvolution):
    def __init__(self, *args, save_averages=False, **kwargs):
        super().__init__(*args, **kwargs)
        if save_averages: self.averages = []
    
    def new_epoch(self, epoch: int):
        # We sample as many architectures as we need
        if epoch < self.population_size:
            model = (
                torch.nn.Module()
            )
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = model.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

            self.population.append(model)
            self._update_history(model)
        else:
            sample = []
            while len(sample) < self.sample_size:
                candidate = np.random.choice(list(self.population))
                sample.append(candidate)

            parent = max(sample, key=lambda x: x.accuracy)

            child = (
                torch.nn.Module()
            )
            child.arch = self.search_space.clone()
            child.arch.mutate(parent.arch, dataset_api=self.dataset_api)
            child.accuracy = child.arch.query(
                self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )

            self.population.append(child)
            self._update_history(child)
            if hasattr(self, 'averages'):
                self.averages.append({'true': np.mean([x.accuracy for x in self.population])})