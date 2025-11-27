from Imports import *


class FilterLength:
    """Filters sentences that are too short or too long.
    3 parameters :
        − min_tokens (default: 3)
        − max_tokens (default: 128)
        − min_chars  (default: 3)
    """

    def __init__(self,
                 min_tokens: int=3,
                 max_tokens: int=128,
                 min_chars: int=3,
                ):
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_chars = min_chars


    def filter_length(self, example: dict[str, Any]) -> bool:
        """Improved filtering for short and long sentences.
        """

        en_text = example["translation"]["en"].strip()
        fr_text = example["translation"]["fr"].strip()

        en_len = len(en_text.split())
        fr_len = len(fr_text.split())

        # Ensure that sentences are not empty and have at least `min_chars` characters
        return (
            self.min_chars <= len(en_text) <= self.max_tokens
            and self.min_chars <= len(fr_text) <= self.max_tokens
            and self.min_tokens <= en_len <= self.max_tokens
            and self.min_tokens <= fr_len <= self.max_tokens
        )


    def check_length_distribution(self,
                                  dataset: Dataset,
                                  dataset_name: str="Dataset",
                                 ) -> None:
        """Checks the distribution of sentence lengths after filtering.
        """

        en_lengths = [len(ex["translation"]["en"].split()) for ex in dataset]
        fr_lengths = [len(ex["translation"]["fr"].split()) for ex in dataset]

        print(f"\n {dataset_name} − Length Check")
        print(f"Total number of sentences: {len(dataset)}")
        print(f"Min length (EN): {min(en_lengths)}, Max length (EN): {max(en_lengths)}")
        print(f"Min length (FR): {min(fr_lengths)}, Max length (FR): {max(fr_lengths)}")

        # Check for out-of-bounds sentences
        errors = [
            (ex["translation"]["en"], ex["translation"]["fr"])
            for ex in dataset
            if len(ex["translation"]["en"].split()) < self.min_tokens
                or len(ex["translation"]["en"].split()) > self.max_tokens
                or len(ex["translation"]["fr"].split()) < self.min_tokens
                or len(ex["translation"]["fr"].split()) > self.max_tokens
        ]

        console = Console()
        if errors:
            console.print(
                f"[bold red]{len(errors)} sentences do NOT meet "
                f"the token limits!"
                f"\n Examples[/bold red]"
            )
            for en, fr in errors[:5]:  # Show a few problematic examples
                console.print(f"[red]EN:[/red] {en}")
                console.print(f"[red]FR:[/red] {fr}")
                console.print("⎯" * 40)
        else:
            console.print(
                f"[bold green]All sentences comply with the {self.min_tokens} "
                f"to {self.max_tokens} token limit.[/bold green]"
            )


def display_random_samples(dataset: Dataset,
                           dataset_name: str="Dataset",
                           n_samples: int=10,
                          ) -> None:
    """Displays random examples for manual dataset inspection.
    """
    print(f"\n{dataset_name}: {n_samples} random samples")

    # Select num_samples random indices from the dataset
    idx = random.sample(range(len(dataset)), min(n_samples, len(dataset)))

    console = Console()
    for i in idx:
        en_text = dataset[i]["translation"].get("en", "")
        fr_text = dataset[i]["translation"].get("fr", "")
        console.print(f"[green]EN[/green]:", en_text)
        console.print(f"[green]FR[/green]:", fr_text)
        console.print(f"⎯" * 40)


####################################################################

from s03_Clean_text_data import (
    ds_train,
    ds_val,
    ds_test,
)


flength = FilterLength()
ds_train = ds_train.filter(flength.filter_length)
ds_val = ds_val.filter(flength.filter_length)
# Not applied on ds_test to avoid biasing the evaluation