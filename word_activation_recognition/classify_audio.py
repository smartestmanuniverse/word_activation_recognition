#coding: utf-8

from activation_defaults import Files_WordRecognition_tflite, labels_activation_phrase
from audio import AudioClassifier
from time import sleep


def activation_callback():
    print('Activation detected')
    sleep(90)
    print('Bye Bye !')
    exit(0)


def run():
    MODEL_FILE = Files_WordRecognition_tflite.model
    LABELS_FILE = Files_WordRecognition_tflite.labels_file

    classifier = AudioClassifier(callback_start_assistant=activation_callback ,model=MODEL_FILE, labels_file=LABELS_FILE)

    while True:
        result = classifier.next(block=True)
        if result is not None:
            label, score = result
            print('Classification:', label, 'score:', score)
    

def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument('model_file', type=str)
    #args = parser.parse_args()

    try:
        run()
    except KeyboardInterrupt:
        print("Interruption par l'utilisateur")
        exit(0)
    except Exception as e:
        print(e)
        exit(1)

if __name__ == '__main__':
    main()
