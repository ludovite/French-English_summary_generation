# Afficher le texte en Markdown
from IPython.display import display, Markdown

def printmd(text: str) -> None:
    display(Markdown(text))


# Convertir un entier avec un séparateur de milliers (espace)
def th_sep(number: int | float) -> str:
    return f"{number:,}".replace(",", " ")