import os
import json
from utils import *
from naslib import utils
from warnings import filterwarnings

filterwarnings('ignore')

name = 'tier1absent'

config = utils.load_config(os.path.join(os.getcwd(), 'config.yaml'))
config.search.epochs = 150
dataset_api = utils.get_dataset_api(search_space=config.search_space, dataset=config.dataset)
tiers = [TrainingSpeedEstimateTier(dataset_api, config, epochs=50, multiplier=0.4),
         QueryFullTrainingTier(config.dataset, dataset_api)]

results, averages = experiment(config, save_averages=True, tiers=tiers)

path = os.path.join(os.getcwd(), 'results', config.search_space, config.dataset, name)
if not os.path.exists(path): os.makedirs(path)
with open(os.path.join(path, 'accuracies.json'), 'w') as f:
    f.write(json.dumps(results))
save_averages(averages, os.path.join(path, 'averages.json'))