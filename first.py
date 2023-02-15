# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 23:42:42 2023

@author: Ivan
"""

import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

CAPUCHIN_FILE = os.path.join('archive', 'Parsed_Capuchinbird_Clips', 'XC3776-3.wav')
NOT_CAPUCHIN_FILE = os.path.join('archive', 'Parsed_Not_Capuchinbird_Clips', 'afternoon-birds-song-in-forest-0.wav')


def load_wav_16k_mono(filename):
    file_contents= tf.io.read_file(filename)
    wav,sample_rate= tf.audio.decode_wav(file_contents, desired_channels=1)
    wav= tf.squeeze(wav, axis=-1)
    sample_ratio=tf.cast(sample_rate, dtype=tf.int64)
    wav=tfio.audio.resample(wav, rate_in=sample_ratio, rate_out=16000)
    return wav

wave= load_wav_16k_mono(CAPUCHIN_FILE)
nwave= load_wav_16k_mono(NOT_CAPUCHIN_FILE)

plt.plot(wave)
plt.plot(nwave)
plt.show()

POS = os.path.join('archive', 'Parsed_Capuchinbird_Clips')
NEG = os.path.join('archive', 'Parsed_Not_Capuchinbird_Clips')

pos=tf.data.Dataset.list_files(POS+'\*.wav')
neg=tf.data.Dataset.list_files(NEG+'\*.wav')

positives= tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives= tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(pos)))))
data= positives.concatenate(negatives)

#print(data.shuffle(1000).as_numpy_iterator().next())

lengths=[]
for file in os.listdir(os.path.join('archive', 'Parsed_Capuchinbird_Clips')):
    tensor_wave = load_wav_16k_mono(os.path.join('archive', 'Parsed_Capuchinbird_Clips', file))
    lengths.append(len(tensor_wave))

print(tf.math.reduce_mean(lengths))
print(tf.math.reduce_min(lengths))
print(tf.math.reduce_max(lengths))

def preprocess(file_path, label):
    wav= load_wav_16k_mono(file_path)
    wav=wav[:48000]
    zero_padding= tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav=tf.concat([zero_padding, wav],0)
    spectrogram= tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram= tf.abs(spectrogram)
    spectrogram= tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label
    

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(16)
data = data.prefetch(8)


train= data.take(36)
test= data.skip(36).take(15)

#samples,labels=train.as_numpy_iterator().next()
#print(labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

model= Sequential()
model.add(Conv2D(16,(3,3),activation="relu",input_shape=(1491,257,1)))
model.add(Conv2D(16,(3,3),activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1,activation="sigmoid"))

model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

model.summary()

hist=model.fit(train, epochs=4, validation_data=test)


plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()


plt.title('Precision')
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['val_precision'], 'b')
plt.show()

plt.title('Recall')
plt.plot(hist.history['recall'], 'r')
plt.plot(hist.history['val_recall'], 'b')
plt.show()

X_test, y_test = test.as_numpy_iterator().next()
yhat = model.predict(X_test)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

def load_mp3_16k_mono(filename):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    res = tfio.audio.AudioIOTensor(filename)
     
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2 
    
    sample_rate = res.rate
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
    return wav

mp3 = os.path.join('archive', 'Forest Recordings', 'recording_00.mp3')

wav = load_mp3_16k_mono(mp3)

audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
samples, index = audio_slices.as_numpy_iterator().next()


def preprocess_mp3(sample, index):
    sample = sample[0]
    zero_padding = tf.zeros([48000] - tf.shape(sample), dtype=tf.float32)
    wav = tf.concat([zero_padding, sample],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram

audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=16000, sequence_stride=16000, batch_size=1)
audio_slices = audio_slices.map(preprocess_mp3)
audio_slices = audio_slices.batch(64)

yhat = model.predict(audio_slices)
yhat = [1 if prediction > 0.5 else 0 for prediction in yhat]

from itertools import groupby

yhat = [key for key, group in groupby(yhat)]
calls = tf.math.reduce_sum(yhat).numpy()

print(calls)

results = {}
for file in os.listdir(os.path.join('archive', 'Forest Recordings')):
    FILEPATH = os.path.join('archive','Forest Recordings', file)
    
    wav = load_mp3_16k_mono(FILEPATH)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(wav, wav, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)
    
    yhat = model.predict(audio_slices)
    
    results[file] = yhat
    
print(results)

class_preds = {}
for file, logits in results.items():
    class_preds[file] = [1 if prediction > 0.99 else 0 for prediction in logits]
class_preds

postprocessed = {}
for file, scores in class_preds.items():
    postprocessed[file] = tf.math.reduce_sum([key for key, group in groupby(scores)]).numpy()
postprocessed


import csv

with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['recording', 'capuchin_calls'])
    for key, value in postprocessed.items():
        writer.writerow([key, value])
