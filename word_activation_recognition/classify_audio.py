#coding: utf-8

import argparse
from audio import classify_audio


def handle_results(label, score):
    print('CALLBACK: ', label, '=>', score)
    return True  # keep listening


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', type=str)
    args = parser.parse_args()

    classify_audio(model=args.model_file, callback=handle_results)

if __name__ == '__main__':
    main()
