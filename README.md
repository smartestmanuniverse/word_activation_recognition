# Word Activation Recognition
> tensorflow project - feature: recognition of activation phrases. ( can be useful to wake : LLM and STT inferences )

## Description

`word_activation_recognition` est une bibliothèque Python pour la détection de phrases d'activation audio. Elle peut être utilisée pour réveiller des inférences LLM (modèles de langage large) et STT (reconnaissance vocale).

## Installation

Pour installer la bibliothèque, vous pouvez utiliser pip :

```bash
pip install word_activation_recognition
```

## Usage

Voici un exemple d'utilisation de la bibliothèque :

```python
from word_activation_recognition import AudioClassifier
from word_activation_recognition.activation_defaults import Files_WordRecognition_tflite

def activation_callback():
    print('Activation detected')

MODEL_FILE = Files_WordRecognition_tflite.model
LABELS_FILE = Files_WordRecognition_tflite.labels_file

classifier = AudioClassifier(callback_start_assistant=activation_callback, model=MODEL_FILE, labels_file=LABELS_FILE)
classifier.run()
```

## Développement

Pour cloner le dépôt et installer les dépendances pour le développement :

```bash
# Clonez le dépôt
git clone https://github.com/smartestmanuniverse/word_activation_recognition.git

# Allez dans le répertoire du projet
cd word_activation_recognition

# Installez les dépendances
pip install -r requirements.txt

# Exécutez les tests
pytest
```

## Tests

Les tests unitaires sont situés dans le répertoire `tests`. Pour exécuter les tests, utilisez la commande suivante :

```bash
pytest
```

## Contribuer

Les contributions sont les bienvenues ! Veuillez soumettre une pull request avec une description détaillée des modifications.

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](./LICENSE) pour plus de détails.

## Auteur

`word_activation_recognition` a été développé par 0x07cb.

## Liens utiles

- [Repository GitHub](https://github.com/smartestmanuniverse/word_activation_recognition)


