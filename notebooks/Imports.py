# STL
from collections import Counter
from dataclasses import dataclass
import html
from pathlib import Path
import random
import re
import string
import time
from typing import Any, Iterable
import unicodedata

# Rich display in console
from rich import print
from rich.console import Console
from rich.table import Table

# Manipulation et entraînement de données
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)

# Spécifique à Jupyter
from ImportsJupyter import *