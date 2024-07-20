#coding: utf-8

import queue
import sys
import contextlib
import json
import numpy as np
import threading
import re
import pyaudio
import ring_buffer
from tflite_support import metadata
import tflite_runtime.interpreter as tflite
import librosa
from multiprocessing import Process
from activation_defaults import Files_WordRecognition_tflite, labels_activation_phrase


def _associated_labels_file(metadata_json):
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
    labels_file = _associated_labels_file(metadata_json)
    labels = displayer.get_associated_file_buffer(labels_file).decode()
    return {i: label for i, label in enumerate(labels.splitlines())}


def read_labels_file(filepath):
    """ 
    Lire les labels depuis un fichier texte et retourner cela comme un dictionnaire.
    Arguments:
    - filepath (str): Le chemin vers le fichier de labels.
    Sortie:
    - dictionnaire de (int, str): Les labels avec les ids comme clés et les descriptions comme valeurs.
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
    return False


class AudioClassifier:
    """Performs classifications with a speech classification model.

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
        self._stop_listening = threading.Event()
        self._thread = threading.Thread(
            target=self.classify_audio,
            kwargs={'callback': self.handle_results, **kwargs},
            daemon=True)
        self._thread.start()
        self._process_assistant = None
        self.callback_start_assistant = callback_start_assistant
        self._audio_stream = None

    def classify_audio(self, model, callback,
                       labels_file=None,
                       inference_overlap_ratio=0.1,
                       buffer_size_secs=2.0,
                       buffer_write_size_secs=0.1,
                       audio_device_index=None):
        """
        Continuously classifies audio samples from the microphone, yielding results
        to your own callback function.
        """
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

        if inference_overlap_ratio < 0.0 or inference_overlap_ratio >= 1.0:
            raise ValueError('inference_overlap_ratio must be in [0.0 .. 1.0)')

        interpreter = tflite.Interpreter(model_path=model)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        waveform_input_index = input_details[0]['index']
        _, num_audio_frames = input_details[0]['shape']
        waveform = np.zeros(num_audio_frames, dtype=np.float32)

        output_details = interpreter.get_output_details()
        scores_output_index = output_details[0]['index']

        def preprocess_audio(audio_data, sr=16000):
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
            self._audio_stream = stream
            keep_listening = True
            while keep_listening:
                while not self._stop_listening.is_set():
                    rb.read(waveform, remove_size=remove_size)
                    interpreter.set_tensor(waveform_input_index, [waveform])
                    interpreter.invoke()
                    scores = interpreter.get_tensor(scores_output_index)
                    scores = np.mean(scores, axis=0)
                    prediction = np.argmax(scores)
                    keep_listening = callback(labels[prediction], scores[prediction])

    def close_audio_stream(self):
        if self._audio_stream is not None:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
            self._audio_stream = None

    def stop_listening(self):
        """Méthode pour arrêter l'écoute."""
        self._stop_listening.set()
        self.close_audio_stream()

    def resume_listening(self):
        """Méthode pour reprendre l'écoute."""
        self._stop_listening.clear()
        self._thread = threading.Thread(
            target=self.classify_audio,
            kwargs={'callback': self.handle_results},
            daemon=True)
        self._thread.start()

    def handle_results(self, label, score):
        label_, score_ = str(label), float(score)
        print('CALLBACK: ', label_, '=>', score_)
        print(f"{type(label_)} => {type(score_)}")

        activation_detected = activation_phrase_detection(label_, score_)
        print(activation_detected)
        if activation_detected:
            if self._process_assistant is None or not self._process_assistant.is_alive():
                self.stop_listening()
                self._process_assistant = Process(target=self.callback_start_assistant, args=())
                self._process_assistant.start()
                self._process_assistant.join()
                self.resume_listening()
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
