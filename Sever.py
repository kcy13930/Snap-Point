"""
musicinformationretrieval.com/realtime_spectrogram.py
PyAudio example: display a live log-spectrogram in the terminal.
For more examples using PyAudio:
    https://github.com/mwickert/scikit-dsp-comm/blob/master/sk_dsp_comm/pyaudio_helper.py
"""
import librosa
import numpy
import pyaudio
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import utils

# Define global variables.
CHANNELS = 1
RATE = 44100
FRAMES_PER_BUFFER = 1024
N_FFT = 4096
SCREEN_WIDTH = 178
ENERGY_THRESHOLD = 0.4

# Choose the frequency range of your log-spectrogram.
F_LO = librosa.note_to_hz('C2')
F_HI = librosa.note_to_hz('C9')
M = librosa.filters.mel(RATE, N_FFT, SCREEN_WIDTH, fmin=F_LO, fmax=F_HI)

p = pyaudio.PyAudio()



load_model = tf.keras.models.load_model('SnapPoint.h5')
load_model.summary()

def mfcc(data):
    batch_size = 0
    d_height = 18
    d_width = 40
    train = numpy.zeros((batch_size, d_height, d_width))
    
    for i in data:
        sample = librosa.feature.mfcc(y=i, sr=44100, n_mfcc=40)
        sample = numpy.expand_dims(sample.T,axis=0)
        data = numpy.concatenate((train, sample),axis=0)
    return data

def test(sample):
    data = mfcc(sample)
    predictions = load_model.predict(data)
    for i in predictions:
        if numpy.armgmax(i)==0:
            print("Fingersnap")
        elif numpy.armgmax(i)==1:
            print("Clap")
        else:
            print("Noise")
            

    
def generate_string_from_audio(audio_data):
    """
    This function takes one audio buffer as a numpy array and returns a
    string to be printed to the terminal.
    """
    # Compute real FFT.
    x_fft = numpy.fft.rfft(audio_data, n=N_FFT)

    # Compute mel spectrum.
    melspectrum = M.dot(abs(x_fft))

    # Initialize output characters to display.
    char_list = [' ']*SCREEN_WIDTH

    for i in range(SCREEN_WIDTH):

        # If there is energy in this frequency bin, display an asterisk.
        if melspectrum[i] > ENERGY_THRESHOLD:
            char_list[i] = '*'

        # Draw frequency axis guidelines.
        elif i % 30 == 29:
            char_list[i] = '|'

    # Return string.
    return ''.join(char_list)

def callback(in_data, frame_count, time_info, status):
    audio_data = numpy.fromstring(in_data, dtype=numpy.float64)
    print( generate_string_from_audio(audio_data) )
    frames = []
    for i in range(0, int(RATE / CHUNK * 0.2)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    test(audio_data)
    return (in_data, pyaudio.paContinue)

stream = p.open(format=pyaudio.paFloat32,
                channels=CHANNELS,
                rate=RATE,
                input=True,   # Do record input.
                output=False, # Do not play back output.
                frames_per_buffer=FRAMES_PER_BUFFER,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    time.sleep(0.100)

stream.stop_stream()
stream.close()

p.terminate()

