from Imports import *


@dataclass
class TextCleaner:
    """Clean different spaces chars and escape sequences in  a text"""
    
    
    allowed_chars: frozenset[str] = frozenset("%€$¥°«»")
    space_chars: frozenset[str] = frozenset({
        '\xa0',      # non-breaking space (NBSP, #160)
        '\u200b',    # zero-width space
        '\u2002',    # en space
        '\u2003',    # em space
        '\u2004',    # three-per-em space
        '\u2005',    # four-per-em space
        '\u2006',    # six-per-em space
        '\u2007',    # figure space
        '\u2008',    # punctuation space
        '\u2009',    # thin space
        '\u200a',    # hair space
        '\u200c',    # zero-width non-joiner
        '\u200d',    # zero-width joiner
        '\u202f',    # narrow no-break space
        '\ufeff',    # BOM UTF-8
    })

    
    def clean_text(self, text: str) -> str:
        """Clean a text by handling all special cases."""
        
        if not isinstance(text, str):
            return ""

        # Decode HTML entities
        text = html.unescape(text)

        # Specific handling of &#160; which may contain spaces
        text = re.sub(r'&\s*#\s*160\s*;', ' ', text)

        # Convert special space characters
        for space_char in self.space_chars:
            text = text.replace(space_char, ' ')

        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)

        # Handle escape sequences
        text = re.sub(r'\\\\([bnrt])', r'\\\1', text)
        text = re.sub(r'\\\\', '', text)

        # Normalize spacing
        text = ' '.join(text.split())

        return text.strip()


def clean_dataset(dataset: Dataset, cleaner: TextCleaner) -> Dataset:
    """Cleans the dataset."""

    def clean_example(example: dict[str, Any]) -> dict[str, Any]:
        if 'translation' in example:
            if 'en' in example['translation']:
                example['translation']['en'] = \
                    cleaner.clean_text(example['translation']['en'])
            if 'fr' in example['translation']:
                example['translation']['fr'] = \
                    cleaner.clean_text(example['translation']['fr'])
        return example

    # Directly modifies the dataset
    dataset = dataset.map(clean_example)
    return dataset


def verify_cleaning(dataset: Dataset, num_examples: int = 5) -> None:
    """Verifies the quality of the cleaning process."""
    console = Console()
    console.print("\n[bold blue]Cleaning Verification[/bold blue]")

    # Search for remaining HTML entities and special characters
    for i, example in enumerate(dataset):
        if i >= num_examples:
            break
    
        en_text = example['translation'].get('en', '')
        fr_text = example['translation'].get('fr', '')

        # Check for the presence of &#160; or similar characters
        if '&#160;' in en_text or '&#160;' in fr_text \
            or '\xa0' in en_text \
            or '\xa0' in fr_text:
            console.print(f"\n[bold red]Example {i+1} still contains special characters:[/bold red]")
            console.print(f"EN: {en_text}")
            console.print(f"FR: {fr_text}")
        else:
            console.print(f"\n[bold green]Example {i+1} is clean:[/bold green]")
            console.print(f"EN: {en_text}")
            console.print(f"FR: {fr_text}")

        console.print("⎯" * 40)


##############################################################

from s02_Split_dataset import ds


cleaner = TextCleaner()
ds_train = clean_dataset(ds['train'], cleaner)
ds_val = clean_dataset(ds['valid'], cleaner)
ds_test = clean_dataset(ds['test'], cleaner)