from Imports import *

# from typing import Iterable
# import string

from s04_Filter_sentence_length import display_random_samples
from s05_Remove_duplicates import (
    ds_train, ds_val, ds_test,
)


class LowerCase():
    """
    Smart lowercase conversion with preservation of acronyms and addresses.
    """

    # Set of acronyms to keep uppercased
    __acronyms = {
        "api", "apache", "bios", "cdn", "cli", "cmd", "cpu", "docker",
        "dns", "dpi", "exe", "firewall", "ftp", "gpu", "gui", "hdmi",
        "html", "http", "https", "ide", "iot", "ip", "ipv4", "ipv6",
        "jpeg", "json", "jwt", "lan", "linux", "mac", "macos", "mariadb",
        "mongodb", "mysql", "nat", "nginx", "nosql", "nfc", "oauth",
        "os", "pci", "pdf", "png", "postgresql", "pytorch", "qos",
        "raid", "ram", "rdbms", "redis", "rfid", "sdk", "scsi", "sata",
        "sdk", "smtp", "ssh", "ssd", "ssl", "tcp", "tcp/ip", "tls",
        "udp", "ui", "ubuntu", "ux", "usb", "usb-c", "vga", "vpn",
        "wan", "windows", "xml", "yaml",
    }

    @classmethod
    def show_acronyms(cls):
        """
        Print a set of acronyms.
        """
        print(cls.__acronyms)

    @classmethod
    def update_acronyms(cls, extra_acronyms: Iterable[str]):
        """
        Add acronyms, from the iterable `extra_acronyms`, to the class.
        """
        for acronym in extra_acronyms:
            cls.__acronyms.add(acronym)

    @classmethod
    def smart_lowercase(cls, text: str) -> str:
        """
        Convert text to lowercase, except for acronyms and email addresses.
        """
        email_pattern = re.compile(r"[\w.-]+@[\w.-]+")
        words = text.split()
        processed_words = []

        for word in words:
            # Check if the word is in the internal acronyms set or is an email address
            # beware of any punctuation symbol at the end of that word
            word_lowered = word.lower()
            if word_lowered.rstrip(string.punctuation) in cls.__acronyms \
            or email_pattern.match(word):
                processed_words.append(word)  # Keep the word as-is
            else:
                processed_words.append(word_lowered)  # Convert to lowercase

        return " ".join(processed_words)

    @classmethod
    def  apply_smart_lowercase(cls, dataset: Dataset) -> Dataset:
        """Applies smart lowercase conversion to the entire dataset."""
        return dataset.map(lambda ex: {
            "translation": {
                "en": cls.smart_lowercase(ex["translation"]["en"]),
                "fr": cls.smart_lowercase(ex["translation"]["fr"])
            }
        })

########################################################################

ds_train = LowerCase().apply_smart_lowercase(ds_train)
ds_val = LowerCase().apply_smart_lowercase(ds_val)
ds_test = LowerCase().apply_smart_lowercase(ds_test)