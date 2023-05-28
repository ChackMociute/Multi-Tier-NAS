import torch
import numpy as np
from abc import ABC, abstractmethod
from naslib.predictors import ZeroCost
from naslib.search_spaces.core import Metric


class Tier(ABC):
    def __init__(self, epochs=None, range=[0, 100], scale=1, multiplier=0):
        self.epochs = epochs
        self.range = range
        self.scale = scale
        # Multiplier only useful for trials for comparing with RE
        self.epoch_multiplier = multiplier
    
    def __call__(self, arch):
        return self.evaluate(arch)

    def normalize(self, score):
        return (self.range[1] - self.range[0]) * self.scale * score + self.range[0]

    @abstractmethod
    def evaluate(self, arch):
        pass


class JaCovTier(Tier, ZeroCost):
    def __init__(self, dataloader, dropoff=1e-2, batches=1, **kwargs):
        ZeroCost.__init__(self, method_type='jacov')
        Tier.__init__(self, **kwargs)
        self.dataloader = dataloader
        self.dropoff = dropoff
        self.num_imgs_or_batches = batches
    
    def evaluate(self, arch):
        if torch.cuda.is_available(): arch.parse()
        return self.normalize(np.exp(self.dropoff*self.query(arch, dataloader=self.dataloader)))


class TrainingSpeedEstimateTier(Tier):
    QUERY_TRANSLATIONS = {
        'cifar10': 'cifar10-valid',
        'cifar100': 'cifar100',
        'ImageNet16-120': 'ImageNet16-120'
    }
    
    def __init__(self, dataset_api, config, dropoff=3e-1, E=1, dataloader=None, multiplier=0.2, **kwargs):
        super().__init__(**kwargs, multiplier=multiplier)
        self.dataset_api = dataset_api
        self.budget = config.search.budget if hasattr(config.search, 'budget') else 10
        self.dataset = config.dataset
        self.dropoff = dropoff
        self.E = E
        if dataloader is not None: self.dataloader = dataloader
        
    def losses(self, arch):
        query_data = arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)
        return [l for l in query_data[self.QUERY_TRANSLATIONS[self.dataset]]['train_losses'][:self.budget]]
    
    def evaluate(self, arch):
        return self.normalize(np.exp(-self.dropoff*sum(self.losses(arch)[-self.E:])))

class QueryFullTrainingTier(Tier):
    def __init__(self, dataset, dataset_api, performance_metric=Metric.VAL_ACCURACY, scale=0.01, multiplier=1, **kwargs):
        super().__init__(**kwargs, scale=scale, multiplier=multiplier)
        self.dataset = dataset
        dataset_api=self.dataset_api = dataset_api
        self.performance_metric = performance_metric
    
    def evaluate(self, arch):
        return self.normalize(arch.query(self.performance_metric, self.dataset, dataset_api=self.dataset_api))