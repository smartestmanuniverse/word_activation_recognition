#coding: utf-8

import contextlib
import json
import numpy as np
import os
import queue
import sys
import threading

import pyaudio
import tflite_runtime.interpreter as tflite

#from pycoral.utils import dataset
import pandas as pd
def read_labels_file(filename):
    # Charger un fichier CSV comme exemple
    dataset = pd.read_csv(f"{filename}")
    return dataset

from tflite_support import metadata

import ring_buffer


@contextlib.contextmanager
def pyaudio_stream(*args, **kwargs):
    """Context manager for the PyAudio stream."""
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(*args, **kwargs)
        try:
            yield stream
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        audio.terminate()



# Function `classify_audio` is used to classify audio data using a TFLite model.
"""
Continuously classifies audio samples from the microphone, yielding results
    to your own callback function.

    inference performed. Although the audio sample size is fixed based on the
    model input size, you can adjust the rate of inference with
    ``inference_overlap_ratio``. A larger overlap means the model runs inference
    more frequently but with larger amounts of sample data shared between
    inferences, which can result in duplicate results.
"""

def classify_audio():
    channels = 1
    
    inference_overlap_ratio = 0.1
    buffer_size_secs = 2.0
    buffer_write_size_secs = 0.1
    audio_device_index = None

    model = 'models/star_trek_activation_phrase_v2.tflite'
    labels_file = 'labels/star_trek_activation_phrase_v2.txt'

    # Paramètres du flux audio
    sample_rate_hz = 16000  # Fréquence d'échantillonnage
    block_size = 1024  # Taille des blocs d'échantillons

    #sample_rate_hz, channels = model_audio_properties(model)

    if labels_file is not None:
        labels = read_labels_file(labels_file)
    else:
        labels = utils.read_labels_from_metadata(model)

    if not model:
        raise ValueError('model must be specified')

    if buffer_size_secs <= 0.0:
        raise ValueError('buffer_size_secs must be positive')

    if buffer_write_size_secs <= 0.0:
        raise ValueError('buffer_write_size_secs must be positive')

    if inference_overlap_ratio < 0.0 or \
       inference_overlap_ratio >= 1.0:
        raise ValueError('inference_overlap_ratio must be in [0.0 .. 1.0)')

    interpreter = tflite.Interpreter(model_path=model)
    interpreter.allocate_tensors()

    # Input tensor
    input_details = interpreter.get_input_details()
    waveform_input_index = input_details[0]['index']
    _, num_audio_frames = input_details[0]['shape']
    waveform = np.zeros(num_audio_frames, dtype=np.float32)

    # Output tensor
    output_details = interpreter.get_output_details()
    scores_output_index = output_details[0]['index']


    # Fonction de prétraitement pour convertir les données audio en Mel spectrogram
    def preprocess_audio(audio_data, sr=16000):
        nonlocal interpreter, input_details
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
        return log_mel_spectrogram.T

    # # Fonction pour traiter le flux audio en temps réel
    # def audio_callback(indata, frames, time, status):
    #     nonlocal interpreter, input_details, output_details
    #     if status:
    #         print(status, flush=True)
    #     # Prétraitement de l'audio
    #     mel_spectrogram = preprocess_audio(indata[:, 0])
    #     input_data = np.expand_dims(mel_spectrogram, axis=0).astype(np.float32)

    #     # Faire une prédiction avec le modèle TFLite
    #     interpreter.set_tensor(input_details[0]['index'], input_data)
    #     interpreter.invoke()
    #     output_data = interpreter.get_tensor(output_details[0]['index'])
        
    #     # Afficher le résultat de la prédiction
    #     print("Résultat de la prédiction:", output_data, flush=True)


    ring_buffer_size = int(buffer_size_secs * sample_rate_hz)
    frames_per_buffer = int(buffer_write_size_secs * sample_rate_hz)
    remove_size = int((1.0 - inference_overlap_ratio) * len(waveform))

    rb = ring_buffer.ConcurrentRingBuffer(
        np.zeros(ring_buffer_size, dtype=np.float32))

    def stream_callback(in_data, frame_count, time_info, status):
        try:
            rb.write(np.frombuffer(in_data, dtype=np.float32), block=False)
        except ring_buffer.Overflow:
            print('WARNING: Dropping input audio buffer', file=sys.stderr)

        return None, pyaudio.paContinue


    try:
        with pyaudio_stream(format=pyaudio.paFloat32,
                            channels=channels,
                            rate=sample_rate_hz,
                            frames_per_buffer=frames_per_buffer,
                            stream_callback=stream_callback,
                            input=True,
                            input_device_index=audio_device_index) as stream:
            keep_listening = True
            while keep_listening:
                rb.read(waveform, remove_size=remove_size)

                interpreter.set_tensor(waveform_input_index, [waveform])
                interpreter.invoke()
                scores = interpreter.get_tensor(scores_output_index)
                scores = np.mean(scores, axis=0)
                prediction = np.argmax(scores)
                keep_listening = callback(labels[prediction], scores[prediction])
                
    except KeyboardInterrupt:
        print("Interruption par l'utilisateur")
        exit(0)
