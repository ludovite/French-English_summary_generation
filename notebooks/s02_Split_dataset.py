from Imports import *
from s01_Load_dataset import data


# Choose a random_state seed
random_seed = 42

# 1. First split: train (90 %) / temp (10 %)
ds_train_temp = data['train'].train_test_split(train_size=0.90, seed=random_seed)

# 2. Second split on temp: validation (75 %) / test (25 %)
ds_temp = ds_train_temp['test'].train_test_split(train_size=0.75, seed=random_seed)

# Put those splits inside a DatasetDict
ds = DatasetDict({
    'train': ds_train_temp['train'],  # 90 %  of the whole dataset
    'valid': ds_temp['train'],        # 7.5 % of the whole dataset
    'test': ds_temp['test']           # 2.5 % of the whole dataset
})

del ds_train_temp
del ds_temp
del data
