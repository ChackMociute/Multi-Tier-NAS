import os
import json
from utils import *
from naslib import utils
from warnings import filterwarnings

filterwarnings('ignore')

name = 'default'

config = utils.load_config(os.path.join(os.getcwd(), 'config.yaml'))
config.dataset = 'ImageNet16-120'

train_loader = utils.get_train_val_loaders(config, mode="train")[0]
dataset_api = utils.get_dataset_api(search_space=config.search_space, dataset=config.dataset)
tiers = [JaCovTier(train_loader, batches=12, dropoff=1.5e-3),
         TrainingSpeedEstimateTier(dataset_api, config, dropoff=4e-1),
         QueryFullTrainingTier(config.dataset, dataset_api)]

results, averages = experiment(config, save_averages=True, tiers=tiers)

path = os.path.join(os.getcwd(), 'results', config.search_space, config.dataset, name)
if not os.path.exists(path): os.makedirs(path)
with open(os.path.join(path, 'accuracies.json'), 'w') as f:
    f.write(json.dumps(results))
save_averages(averages, os.path.join(path, 'averages.json'))