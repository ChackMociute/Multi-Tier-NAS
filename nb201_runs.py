import json
import pandas as pd
from tqdm import tqdm
from naslib import utils
from naslib.search_spaces import NasBench201SearchSpace
from naslib.defaults.trainer import Trainer
from MultiTierNAS import MultiTierNAS
from RegularizedEvolution import RegularizedEvolution
from naslib.optimizers import RandomSearch
from naslib.search_spaces.core import Metric


config = utils.load_config('D:/Desktop/(3)Thesis/Multi-Tier-NAS/config.yaml')
dataset_api = utils.get_dataset_api(search_space=config.search_space, dataset=config.dataset)
search_space = NasBench201SearchSpace()
train_loader, _, _, _, _ = utils.get_train_val_loaders(config, mode="train")
epochs = config.search.epochs

results = []
results_dict = list()
for i, seed in tqdm(enumerate(eval(str(config.seeds)))):
    results.append(dict())
    config.search.epochs = epochs
    config.search.seed = seed
    
    mtnas = MultiTierNAS(config, save_averages=True)
    mtnas.adapt_search_space(search_space, dataset_api=dataset_api)
    mtnas.set_default_tiers(dataloader=train_loader)
    trainer_mtnas = Trainer(mtnas, config)
    trainer_mtnas.name = 'mtnas'
    
    config.search.epochs = epochs // len(mtnas.tiers) + 20
    re = RegularizedEvolution(config, save_averages=True)
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
        
    results_dict.append({'mtnas': mtnas.averages, 're': re.averages})

with open('nb201_runs.json', 'w') as f:
    f.write(json.dumps(results))

df = pd.DataFrame.from_dict({(key, i, k): [x[k] for x in val if k in x.keys()] for i, res in enumerate(results_dict) for key, val in res.items() for k in ['estm', 'true']}, orient='index').T
df.columns = pd.MultiIndex.from_tuples(df.columns).rename(['algorithm', 'seed'], level=[0, 1])
df.index.rename('epoch', inplace=True)
with open('nb201_averages.json', 'w') as f:
    f.write(df.to_json())