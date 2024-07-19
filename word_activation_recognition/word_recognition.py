#!/usr/bin/env python3
#coding: utf-8

#from aiymakerkit import audio

import numpy as np
import pyaudio
import sounddevice as sd
import librosa
import tflite_runtime.interpreter as tflite

class labels_activation_phrase:
    label_0 = "0 Background Noise"
    label_1 = "1 annuler"
    label_2 = "2 ordinateur"
    # ##############################
    min_score_label_2 = float(0.3)
    min_score_label_1 = float(0.3)
    min_score_label_3 = float(0.3)

def activation_callback(label, score):
    print(label, '=>', score)
    print('Activation detected')
    exit(0)

def activation_phrase_detection(label, score):
    if label == labels_activation_phrase.label_2:
        print(f"{label} is detected...")
        if score > labels_activation_phrase.min_score_label_2:
            print(f"ok now the label '{label}' pass with the score of : '{score}'")
            return True
        else:
            return False
    else:
        return False
    return False

def choice_bin_response_detection(label, score):
    if label == labels_activation_phrase.label_1:
        if score > labels_activation_phrase.min_score_label_1:
            return True
        else:
            return False
    elif label == labels_activation_phrase.label_3:
        if score > labels_activation_phrase.min_score_label_3:
            return True
        else:
            return False
    else:
        return False
    return False

def audio_activation(activation_callback):
    def handle_results(label_, score_):
        label, score = str(label_), float(score_)
        print('CALLBACK: ', label, '=>', score)
        print(f"{type(label)} => {type(score)}")

        activation_detected = activation_phrase_detection(label, score)
        print(activation_detected)
        if activation_detected:
            activation_callback(label, score)
            return False 
        return True

    model_file = 'models/star_trek_activation_phrase_v2.tflite'

    audio.classify_audio(
                            model=model_file, 
                            callback=handle_results
                        )


# ##############################
# 
# ##############################
# def preprocess_audio(audio_path):
#    # Charger l'audio et convertir en Mel spectrogram (à adapter selon ton besoin)
#    mel_spectrogram = generate_mel_spectrogram(audio_path) # Fonction personnalisée
#    input_data = np.array(mel_spectrogram, dtype=np.float32)
#    return input_data


def generate_mel_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=16000)
    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
    return log_mel_spectrogram.T




# ##############################
def main():
    model_file = 'models/star_trek_activation_phrase_v2.tflite'

    # Charger le modèle
    interpreter = tflite.Interpreter(model_path=model_file)
    interpreter.allocate_tensors()

    # Obtenir les détails des entrées et sorties
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Paramètres du flux audio
    sample_rate = 16000  # Fréquence d'échantillonnage
    block_size = 1024  # Taille des blocs d'échantillons


    # Fonction de prétraitement pour convertir les données audio en Mel spectrogram
    def preprocess_audio(audio_data, sr=16000):
        nonlocal interpreter, input_details
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram.T

    # Fonction pour traiter le flux audio en temps réel
    def audio_callback(indata, frames, time, status):
        nonlocal interpreter, input_details, output_details
        if status:
            print(status, flush=True)
        # Prétraitement de l'audio
        mel_spectrogram = preprocess_audio(indata[:, 0])
        input_data = np.expand_dims(mel_spectrogram, axis=0).astype(np.float32)

        # Faire une prédiction avec le modèle TFLite
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        
        # Afficher le résultat de la prédiction
        print("Résultat de la prédiction:", output_data, flush=True)
 
    try:
        # Démarrer le flux audio avec SoundDevice
        with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=block_size):
            print("En écoute... Appuyez sur Ctrl+C pour arrêter.", flush=True)
            while True:
                pass
    except KeyboardInterrupt:
        print("Interruption par l'utilisateur")
        exit(0)
    
    



if __name__ == '__main__':
    try:
        main()
        # audio_activation(activation_callback=activation_callback)
    except KeyboardInterrupt:
        print('Bye')
        exit(0)
    except Exception as e:
        print('ERROR:', e)
        exit(1)
    finally:
        print('Bye')
        exit(0)

