import numpy as np
from naslib.search_spaces.core import Metric

class TrainingSpeedEstimate():
    QUERY_TRANSLATIONS = {
        'cifar10': 'cifar10-valid',
        'cifar100': 'cifar100',
        'ImageNet16-120': 'ImageNet16-120'
    }
    
    def __init__(self, dataloader, budget=10, dataset=None, dataset_api=None, query_nb201=False):
        self.dataloader = dataloader
        self.budget = budget
        self.query_nb201 = query_nb201
        self.dataset = dataset
        self.dataset_api = dataset_api
    
    def evaluate(self, arch):
        query_data = arch.query(Metric.RAW, self.dataset, dataset_api=self.dataset_api)
        return 100*np.exp(-1e-2*sum([l for l in query_data[self.QUERY_TRANSLATIONS[self.dataset]]['train_losses'][:self.budget]]))