import torch
import logging
import warnings
import numpy as np
from naslib import utils
from naslib.optimizers import RegularizedEvolution
from naslib.utils.log import log_every_n_seconds
from tiers import JaCovTier, TrainingSpeedEstimateTier, QueryFullTrainingTier

logger = logging.getLogger(__name__)


class MultiTierNAS(RegularizedEvolution):
    def __init__(self, *args, save_averages=False, reevaluate=False, **kwargs):
        super().__init__(*args, **kwargs)
        if save_averages: self.averages = []
        self.reevaluate = reevaluate
    
    def find_epochs_per_tier(self):
        epochs_per_tier = np.asarray([
            np.nan if tier.epochs is None else tier.epochs for tier in self.tiers
        ])

        designated_epochs = epochs_per_tier[~np.isnan(epochs_per_tier)].sum()
        undesignated_tiers = np.isnan(epochs_per_tier).sum()
        if designated_epochs > self.epochs - undesignated_tiers:
            raise ValueError("Tiers have too many epochs")
        
        epochs_per_undesignated_tiers = (self.epochs - designated_epochs) /\
            undesignated_tiers if undesignated_tiers > 0 else 0
        self.epochs_per_tier = np.nan_to_num(epochs_per_tier, copy=False,
                                             nan=epochs_per_undesignated_tiers
                                            ).astype(int).cumsum()
        if self.epochs_per_tier[-1] != self.epochs:
            if self.tiers[-1].epochs is not None:
                warnings.warn("The sum of all tier epochs does not match " +
                              "the total number of epochs. Final tier will " +
                              "be used to evaluate more than specified.")
            self.epochs_per_tier[-1] = self.epochs

    def set_tiers(self, tiers):
        self.tiers = tiers
        self.evaluator_ind = 0
        self.evaluator = self.tiers[self.evaluator_ind]
        self.find_epochs_per_tier()

    # Call only after adapting search space
    def set_default_tiers(self, dataloader=None):
        if dataloader is None: dataloader = utils.get_train_val_loaders(self.config, mode="train")[0]
        self.set_tiers([JaCovTier(dataloader),
                        TrainingSpeedEstimateTier(self.dataset_api, self.config),
                        QueryFullTrainingTier(self.dataset, self.dataset_api)])
    
    def before_training(self):
        logger.info("Start sampling architectures to fill the population")

        for _ in range(self.population_size):
            model = torch.nn.Module()
            model.arch = self.search_space.clone()
            model.arch.sample_random_architecture(dataset_api=self.dataset_api)
            model.accuracy = self.evaluator(model.arch)

            self.population.append(model)
            if self.evaluator == self.tiers[-1]: self._update_history(model)
            log_every_n_seconds(
                logging.INFO, "Population size {}".format(len(self.population))
            )
        if hasattr(self, 'averages'):
            self.averages.append({'estm': np.mean([x.accuracy for x in self.population]),
                                'true': np.mean([x.arch.query(self.performance_metric,
                                                                self.dataset, dataset_api=self.dataset_api)
                                                                for x in self.population])})

    def reevaluate_population(self):
        for model in self.population:
            model.accuracy = self.evaluator(model.arch)
            if self.evaluator == self.tiers[-1]: self._update_history(model)
    
    def update_evaluator(self):
        self.evaluator_ind += 1
        try: self.evaluator = self.tiers[self.evaluator_ind]
        except IndexError: self.evaluator = self.tiers[-1]
        if self.reevaluate: self.reevaluate_population()

    def new_epoch(self, epoch: int):
        if epoch in self.epochs_per_tier: self.update_evaluator()

        sample = np.random.choice(list(self.population), self.sample_size)
        parent = max(sample, key=lambda x: x.accuracy)

        child = torch.nn.Module()
        child.arch = self.search_space.clone()
        child.arch.mutate(parent.arch, dataset_api=self.dataset_api)
        child.accuracy = self.evaluator(child.arch)

        self.population.append(child)
        if self.evaluator == self.tiers[-1]: self._update_history(child)

        if hasattr(self, 'averages'):
            self.averages.append({'estm': np.mean([x.accuracy for x in self.population]),
                                  'true': np.mean([x.arch.query(self.performance_metric,
                                                                self.dataset, dataset_api=self.dataset_api)
                                                                for x in self.population])})
    
    def get_final_architecture(self):
        try: return max(self.history, key=lambda x: x.accuracy).arch
        except ValueError: return max(self.population, key=lambda x: x.accuracy).arch