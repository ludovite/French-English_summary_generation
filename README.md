


## Ressources utilisées

- Modèle : [mbart](https://huggingface.co/facebook/mbart-large-50) est un modèle séquentiel multilingue. Pré-entraîné, il peut réaliser des traductions automatiques pour 50 langues, après fine-tuning.

- Dataset en Anglais : [xsum](https://huggingface.co/datasets/EdinburghNLP/xsum) est un jeu de données regroupant environ 200 000 articles de la BBC des années 2010. Orienté « eXtreme Summarization », le jeu de données associe à chaque article un résumé d’une phrase.

- Dataset en Français : [mlsum](https://huggingface.co/datasets/reciTAL/mlsum) contient plus d’un million d’articles avec résumé, tirés de journaux en ligne, dans cinq langues différentes.


## Phase d’entraînement

Le code des différentes étapes est stocké dans le répertoire [notebooks](./notebooks/) du dépôt.


## Performances du modèle

L’analyse des performances obtenues est codée dans le fichier [](./notebooks/09_Generate_evaluate_translations.ipynb) du dépôt.


## Application
