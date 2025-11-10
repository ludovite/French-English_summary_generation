# STL
from collections import Counter
from dataclasses import dataclass
import html
import random
import re
import string
from typing import Any, Iterable
import unicodedata

# Rich display in console
from rich import print
from rich.console import Console
from rich.table import Table

# Manipulation de données
from datasets import load_dataset, Dataset, DatasetDict
import numpy as np
from sklearn.model_selection import train_test_split

# Spécifique à Jupyter
from ImportsJupyter import *