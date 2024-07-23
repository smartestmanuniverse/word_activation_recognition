# coding: utf-8
import pkg_resources

class Files_WordRecognition_tflite:
    model = pkg_resources.resource_filename(__name__, 'models/star_trek_activation_phrase_v2.tflite')
    labels_file = pkg_resources.resource_filename(__name__, 'labels/star_trek_activation_phrase_v2.txt')

class labels_activation_phrase:
    label_0 = "Background Noise"
    label_1 = "annuler"
    label_2 = "ordinateur"
    # ##############################
    min_score_label_1 = float(0.3)
    min_score_label_2 = float(0.3)
    min_score_label_3 = float(0.3)
