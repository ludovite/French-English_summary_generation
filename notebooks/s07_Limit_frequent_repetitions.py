from Imports import *
from s06_Smart_lowercase_conversion import (
    ds_train,
    ds_val,
    ds_test,
)

def balance_dataset(dataset: Dataset, max_count: int=5) -> Dataset:
    """
    Dataset balancing function.
    """
    counts_en = Counter()  # Counter for English sentences
    counts_fr = Counter()  # Counter for French sentences
    balanced_data = []

    for example in dataset:
        en_text = example["translation"]["en"]
        fr_text = example["translation"]["fr"]

        # Ensure both English and French sentences do not exceed max_count
        if counts_en[en_text] < max_count \
        and counts_fr[fr_text] < max_count:
            balanced_data.append(example)
            counts_en[en_text] += 1
            counts_fr[fr_text] += 1

    # 📌 Return a Hugging Face Dataset from the balanced list
    return Dataset.from_list(balanced_data)