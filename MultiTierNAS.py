import torch
import logging
import numpy as np
from naslib import utils
from naslib.optimizers import RegularizedEvolution
from naslib.predictors import ZeroCost
from naslib.utils.log import log_every_n_seconds
from TSE import TrainingSpeedEstimate

logger = logging.getLogger(__name__)


class MTNAS(RegularizedEvolution):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_default_tiers(self, dataloader=None):
        # TODO: add customizable tiers
        if dataloader is None: dataloader = utils.get_train_val_loaders(self.config, mode="train")[0]
        self.tiers = self.default_tiers(dataloader)
        self.evaluator_ind = 0
        self.evaluator = self.tiers[self.evaluator_ind]
    
    def default_tiers(self, dataloader):
        zc = ZeroCost(method_type='jacov')
        tier1 = lambda arch: 100*np.exp(0.001*zc.query(arch, dataloader=dataloader))
        
        tse = TrainingSpeedEstimate(dataloader, budget=self.config.search.budget, dataset=self.dataset, dataset_api=self.dataset_api)
        tier2 = lambda arch: tse.evaluate(arch)

        tier3 = lambda arch: arch.query(
            self.performance_metric, self.dataset, dataset_api=self.dataset_api
            )
        return [tier1, tier2, tier3]
    
    def before_training(self):
        logger.info("Start sampling architectures to fill the population")

        for _ in range(self.population_size):
            model = torch.nn.Module()
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = self.evaluator(model.arch)

            self.population.append(model)
            self._update_history(model)
            log_every_n_seconds(
                logging.INFO, "Population size {}".format(len(self.population))
            )
        
    def new_epoch(self, epoch: int):
        if epoch > 0 and epoch % int(self.epochs/len(self.tiers)) == 0:
            self.evaluator_ind += 1
            self.evaluator = self.tiers[self.evaluator_ind]
            for model in self.population:
                model.accuracy = self.evaluator(model.arch)

        sample = []
        while len(sample) < self.sample_size:
            candidate = np.random.choice(list(self.population))
            sample.append(candidate)

        parent = max(sample, key=lambda x: x.accuracy)

        child = torch.nn.Module()
        child.arch = self.search_space.clone()
        child.arch.mutate(parent.arch, dataset_api=self.dataset_api)
        child.accuracy = self.evaluator(child.arch)

        self.population.append(child)
        self._update_history(child)