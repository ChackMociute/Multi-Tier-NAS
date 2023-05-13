import os
import pandas as pd
from tqdm import tqdm
from naslib import utils
from naslib.defaults.trainer import Trainer
from naslib.optimizers import RandomSearch
from naslib.search_spaces import NasBench101SearchSpace, NasBench201SearchSpace, NasBench301SearchSpace
from naslib.search_spaces.core import Metric

import sys
sys.path.append('../')
from MultiTierNAS import MultiTierNAS
from RegularizedEvolution import RegularizedEvolution
from tiers import JaCovTier, TrainingSpeedEstimateTier, QueryFullTrainingTier


def get_search_space(search_space):
    if search_space == 'nasbench101': return NasBench101SearchSpace()
    elif search_space == 'nasbench201': return NasBench201SearchSpace()
    elif search_space == 'nasbench301': return NasBench301SearchSpace()

def experiment(config, save_averages=False, reevaluate=False, tiers=None):
    config.data = os.path.join(os.path.dirname(os.getcwd()), 'naslib/naslib/data')
    search_space = get_search_space(config.search_space)
    dataset_api = utils.get_dataset_api(search_space=config.search_space, dataset=config.dataset)
    if tiers is None: train_loader = utils.get_train_val_loaders(config, mode="train")[0]
    epochs = config.search.epochs

    results = list()
    if save_averages: averages = list()
    for i, seed in tqdm(enumerate(eval(str(config.seeds)))):
        results.append(dict())
        config.search.epochs = epochs
        config.search.seed = seed
        
        # Instantiate optimizers and respective trainers
        mtnas = MultiTierNAS(config, save_averages=save_averages, reevaluate=reevaluate)
        mtnas.adapt_search_space(search_space, dataset_api=dataset_api)
        if tiers is None: mtnas.set_default_tiers(dataloader=train_loader)
        else: mtnas.set_tiers(tiers)
        trainer_mtnas = Trainer(mtnas, config)
        trainer_mtnas.name = 'mtnas'
        
        # Reduce number of epochs to represent equivalent running time
        config.search.epochs = epochs // len(mtnas.tiers) + 200 // config.search.budget
        re = RegularizedEvolution(config, save_averages=save_averages)
        re.adapt_search_space(search_space, dataset_api=dataset_api)
        trainer_re = Trainer(re, config)
        trainer_re.name = 're'
        
        rs = RandomSearch(config)
        rs.adapt_search_space(search_space, dataset_api=dataset_api)
        trainer_rs = Trainer(rs, config)
        trainer_rs.name = 'rs'
        
        for trainer in [trainer_mtnas, trainer_re, trainer_rs]:
            trainer.search()
            results[i][trainer.name] = trainer.evaluate(dataset_api=dataset_api, metric=Metric.VAL_ACCURACY)
            
        if save_averages: averages.append({'mtnas': mtnas.averages, 're': re.averages})
    if save_averages: return results, averages
    return results

def save_averages(averages, path):
    df = pd.DataFrame.from_dict({(key, i, k): [x[k] for x in val if k in x.keys()] for i, res in enumerate(averages) for key, val in res.items() for k in ['estm', 'true']}, orient='index').T
    df.columns = pd.MultiIndex.from_tuples(df.columns).rename(['algorithm', 'seed'], level=[0, 1])
    df.index.rename('epoch', inplace=True)
    with open(path, 'w') as f:
        f.write(df.to_json())