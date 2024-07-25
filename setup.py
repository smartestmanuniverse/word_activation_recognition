from setuptools import setup, find_packages

setup(
    name='word_activation_recognition',
    version='0.1.2',
    packages=find_packages(),
    install_requires=[
        'tflite-runtime',
        'tflite-support',
        'pandas',
        'pyaudio',
        'numpy<2.0.0',
        'librosa',
        'sounddevice',
        'soundfile',
    ],
    package_data={
        'word_activation_recognition': ['models/*.tflite', 'labels/*.txt'],
    },
    entry_points={
        'console_scripts': [
            'word_activation_recognition=word_activation_recognition.classify_audio:main',
        ],
    },
    author='0x07cb',
    author_email='83157348+0x07CB@users.noreply.github.com',
    license='MIT',
    description='A library for audio keyword detection and processing with TensorFlow Lite models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/smartestmanuniverse/word_activation_recognition',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
)



