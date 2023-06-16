import os
import json
import numpy as np
from utils import *
from naslib import utils
from warnings import filterwarnings

filterwarnings('ignore')

name = 'closeness'

config = utils.load_config(os.path.join(os.getcwd(), 'config.yaml'))
config.seeds = 'range(10)'
train_loader = utils.get_train_val_loaders(config, mode="train")[0]
dataset_api = utils.get_dataset_api(search_space=config.search_space, dataset=config.dataset)

for t1 in tqdm(np.linspace(1e-2, 1e-3, 6)):
    for t2 in tqdm(np.linspace(5e-1, 5e-2, 6)):
        tiers = [JaCovTier(train_loader, dropoff=t1),
                TrainingSpeedEstimateTier(dataset_api, config, dropoff=t2),
                QueryFullTrainingTier(config.dataset, dataset_api)]
        results, averages = experiment(config, save_averages=True, tiers=tiers)

        path = os.path.join(os.getcwd(), 'results', config.search_space, config.dataset, name, f"{t1},{t2}")
        if not os.path.exists(path): os.makedirs(path)
        with open(os.path.join(path, 'accuracies.json'), 'w') as f:
            f.write(json.dumps(results))
        save_averages(averages, os.path.join(path, 'averages.json'))