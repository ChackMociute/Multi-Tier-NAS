import json
from tqdm import tqdm
from naslib import utils
from naslib.search_spaces import NasBench201SearchSpace
from naslib.defaults.trainer import Trainer
from MultiTierNAS import MTNAS
from naslib.optimizers import RegularizedEvolution, RandomSearch
from naslib.search_spaces.core import Metric

config = utils.load_config('D:/Desktop/(3)Thesis/Multi-Tier-NAS/config.yaml')
dataset_api = utils.get_dataset_api(search_space=config.search_space, dataset=config.dataset)
search_space = NasBench201SearchSpace()
train_loader, _, _, _, _ = utils.get_train_val_loaders(config, mode="train")
epochs = config.search.epochs

results = []
for i, seed in tqdm(enumerate(eval(str(config.seeds)))):
    results.append(dict())
    config.search.epochs = epochs
    config.search.seed = seed
    
    mtnas = MTNAS(config)
    mtnas.adapt_search_space(search_space, dataset_api=dataset_api)
    mtnas.set_default_tiers(dataloader=train_loader)
    trainer_mtnas = Trainer(mtnas, config)
    trainer_mtnas.name = 'mtnas'
    
    config.search.epochs = epochs // len(mtnas.tiers) + 20
    re = RegularizedEvolution(config)
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

with open('nb201_runs.json', 'w') as f:
    f.write(json.dumps(results))