from Imports import *
from s07_Limit_frequent_repetitions import (
    ds_train, ds_val, ds_test,
    balance_dataset,
)


ds_train = balance_dataset(ds_train)