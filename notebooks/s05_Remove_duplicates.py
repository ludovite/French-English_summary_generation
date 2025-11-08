from Imports import *
from s04_Filter_sentence_length import (
    ds_train,
    ds_val,
    ds_test,
)


def remove_all_repeated_sentences(dataset: Dataset,
                                  dataset_name: str="dataset",
                                 ) -> Dataset:
    """
    Completely removes sentences that appear more than once in the dataset.
    """

    # Count occurrences of sentence pairs
    translations = [
        (
            ex["translation"]["en"].strip().lower(),
            ex["translation"]["fr"].strip().lower()
        )
        for ex in dataset
    ]
    counts = Counter(translations)

    # Remove repeated sentence pairs
    unique_dataset = [
        ex for ex in dataset
        if counts[(ex["translation"]["en"].strip().lower(),
                   ex["translation"]["fr"].strip().lower()
                 )] == 1
    ]

    print(
        f"Number of examples after strict removal of all repeated sentences "
        f"in {dataset_name.lower()}: "
        f"{len(unique_dataset)}"
    )

    return dataset.select(range(len(unique_dataset)))  # Updated dataset


def check_duplicates(dataset: Dataset,
                     dataset_name="Dataset",
                    ) -> None:
    """
    Checks for duplicate sentence pairs in the dataset.
    """

    translations = [
        (ex["translation"]["en"], ex["translation"]["fr"])
        for ex in dataset
    ]
    count = Counter(translations)

    # Detect duplicates
    duplicates = {
        pair: freq
        for pair, freq in count.items()
        if freq > 1
    }

    # Report
    print(f"\n{dataset_name.capitalize()} − Duplicate Check")
    print(f"Total number of examples: {len(dataset)}")
    print(f"Number of unique examples: {len(count)}")

    if duplicates:
        print(f"{len(duplicates)} sentence pairs are duplicated!")
        print("Examples of the most frequent duplicates:")
        for (en, fr), freq in list(duplicates.items())[:5]:  # Show the top 5 duplicates
            print(f"({freq}x) EN: {en}")
            print(f"({freq}x) FR: {fr}")
            print("⎯" * 40)
    else:
        print("No duplicated sentence pairs found.")


#####################################################

ds_train = remove_all_repeated_sentences(ds_train, "Train dataset")
ds_val = remove_all_repeated_sentences(ds_val, "Validation dataset")
# Duplicates are not removed in test set because we want to test even frequent phrases.