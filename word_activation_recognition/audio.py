#coding: utf-8

import contextlib
import json
import numpy as np
import queue
import sys
import threading
import re
import json

import tflite_runtime.interpreter as tflite
from tflite_support import metadata
import pyaudio
import pandas as pd

import ring_buffer

from activation_defaults import Files_WordRecognition_tflite, labels_activation_phrase

import librosa
from multiprocessing import Process


def _associcated_labels_file(metadata_json):
    for ot in metadata_json['subgraph_metadata'][0]['output_tensor_metadata']:
      if 'associated_files' in ot:
        for af in ot['associated_files']:
          if af['type'] in ('TENSOR_AXIS_LABELS', 'TENSOR_VALUE_LABELS'):
            return af['name']
    raise ValueError('Model metadata does not have associated labels file')


def read_labels_from_metadata(model):
    """Read labels from the model file metadata.

    Args:
        model (str): Path to the ``.tflite`` file.
    Returns:
        A dictionary of (int, string), mapping label ids to text labels.
    """
    displayer = metadata.MetadataDisplayer.with_model_file(model)
    metadata_json = json.loads(displayer.get_metadata_json())
    labels_file = _associcated_labels_file(metadata_json)
    labels = displayer.get_associated_file_buffer(labels_file).decode()
    return {i: label for i, label in enumerate(labels.splitlines())}



def read_labels_file(filepath):
    """ 
    
    Lire les labels depuis un fichier texte et retourner cela comme un dictionnaire

    ----------

    Cette fonction supporte les fichiers de labels avec les formats suivants:

    - Chaques ligne contient un id et la description separe par un espace ou une virgule
    ex: ``0:cat`` or ``0 cat``.
    - Chaques ligne contient seulement la description. Les ids sont assignes automatiquement baser sur le numero de ligne. 

    ----------

    Arguments:
    - filepath (str): Le chemin vers le fichier de labels.

    Sortie:
    - dictionnaire de (int, str): Les labels avec les ids comme cles et les descriptions comme valeurs.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ret = {}
    for num_row, content in enumerate(lines):
        pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
        if len(pair) == 2 and pair[0].strip().isdigit():
            ret[int(pair[0])] = pair[1].strip()
        else:
            ret[num_row] = content.strip()
    return ret 

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

def model_audio_properties(model_file):
    """
    Returns the audio sample rate and number of channels that must be used with
    the given model (as a tuple in that order).
    """
    displayer = metadata.MetadataDisplayer.with_model_file(model_file)
    metadata_json = json.loads(displayer.get_metadata_json())
    if metadata_json['name'] != 'AudioClassifier':
        raise ValueError('Model must be an audio classifier')
    props = metadata_json['subgraph_metadata'][0]['input_tensor_metadata'][0]['content']['content_properties']
    return int(props['sample_rate']), int(props['channels'])


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

def classify_audio(model, callback,
                   labels_file=None,
                   inference_overlap_ratio=0.1,
                   buffer_size_secs=2.0,
                   buffer_write_size_secs=0.1,
                   audio_device_index=None):
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
    channels = 1
    
    inference_overlap_ratio = 0.1
    buffer_size_secs = 2.0
    buffer_write_size_secs = 0.1
    audio_device_index = None

    sample_rate_hz, channels = model_audio_properties(model)

    if labels_file is not None:
        labels = read_labels_file(labels_file)
    else:
        labels = read_labels_from_metadata(model)


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
            # ###### Prétraitement de l'audio
            interpreter.set_tensor(waveform_input_index, [waveform])
            interpreter.invoke()
            scores = interpreter.get_tensor(scores_output_index)
            scores = np.mean(scores, axis=0)
            prediction = np.argmax(scores)
            keep_listening = callback(labels[prediction], scores[prediction])

class AudioClassifier:
    """Performs classifications with a speech classification model.

    This is intended for situations where you want to write a loop in your code
    that fetches new classification results in each iteration (by calling
    :func:`next()`). If you instead want to receive a callback each time a new
    classification is detected, instead use :func:`classify_audio()`.

    Args:
        model (str): Path to a ``.tflite`` file.
        labels_file (str): Path to a labels file (required only if the model
            does not include metadata labels). If provided, this overrides the
            labels file provided in the model metadata.
        inference_overlap_ratio (float): The amount of audio that should overlap
            between each sample used for inference. May be 0.0 up to (but not
            including) 1.0. For example, if set to 0.5 and the model takes a
            one-second sample as input, the model will run an inference every
            half second, or if set to 0, it will run once each second.
        buffer_size_secs (float): The length of audio to hold in the audio
            buffer.
        buffer_write_size_secs (float): The length of audio to capture into the
            buffer with each sampling from the microphone.
        audio_device_index (int): The audio input device index to use.
    """

    def __init__(self, callback_start_assistant, **kwargs):
        self._queue = queue.Queue()
        self._thread = threading.Thread(
            target=classify_audio,
            kwargs={'callback': self.handle_results, **kwargs},
            daemon=True)
        self._thread.start()
        self._process_assistant = Process(target=callback_start_assistant, args=())
        self.callback_start_assistant = callback_start_assistant

    def _callback(self, label, score):
        self._queue.put((label, score))
        return True
    
    def handle_results(self, label, score):
        label_, score_ = str(label), float(score)
        print('CALLBACK: ', label_, '=>', score_)
        print(f"{type(label_)} => {type(score_)}")

        activation_detected = activation_phrase_detection(label_, score_)
        print(activation_detected)
        if activation_detected:
            if not self._process_assistant.is_alive():
                self._process_assistant = Process(target=self.callback_start_assistant, args=())
                self._process_assistant.start()
                self._process_assistant.join()
            #activation_callback(label_, score_)
            return False 
        return True

    def next(self, block=True):
        """
        Returns a single speech classification.

        Each time you call this, it pulls from a queue of recent
        classifications. So even if there are many classifications in a short
        period of time, this always returns them in the order received.

        Args:
            block (bool): Whether this function should block until the next
                classification arrives (if there are no queued classifications).
                If False, it always returns immediately and returns None if the
                classification queue is empty.
        """
        try:
            result = self._queue.get(block)
            self._queue.task_done()
            return result
        except queue.Empty:
            return None
