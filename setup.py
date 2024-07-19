from setuptools import setup, find_packages

setup(
    name='word_activation_recognition',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        # Liste des dépendances de votre librairie
        'tflite-runtime',
        'tflite-support',
        'pandas',
        'pyaudio',
        'numpy<2.0.0',
        'librosa',
        'sounddevice',
        'soundfile'
    ],
    # Ajoutez ici d'autres métadonnées pertinentes
    author='0x07cb',
    
)