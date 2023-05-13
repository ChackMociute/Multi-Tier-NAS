import os
import json
from utils import *
from naslib import utils
from warnings import filterwarnings

filterwarnings('ignore')

name = 'reevaluate'

config = utils.load_config(os.path.join(os.getcwd(), 'config.yaml'))
results, averages = experiment(config, save_averages=True, reevaluate=True)

path = os.path.join(os.getcwd(), 'results', config.search_space, config.dataset, name)
if not os.path.exists(path): os.makedirs(path)
with open(os.path.join(path, 'accuracies.json'), 'w') as f:
    f.write(json.dumps(results))
save_averages(averages, os.path.join(path, 'averages.json'))