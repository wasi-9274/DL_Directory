import numpy as np
import wave
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.io import wavfile

mix_1_wave = wave.open('/home/wasi/ML_FOLDER/DSND_Term1-master/lessons/Unsupervised/5_ICA/ICA mix 1.wav', 'r')

print(mix_1_wave.getparams())

signal_1_raw = mix_1_wave.readframes(-1)
# print(signal_1_raw)
signal_1 = np.fromstring(signal_1_raw, 'Int16')
# print(signal_1)
'length: ', len(signal_1), 'first 100 elements: ', signal_1[:100]

fs = mix_1_wave.getframerate()
timing = np.linspace(0, len(signal_1)/fs, num=len(signal_1))

plt.figure(figsize=(12, 22))
plt.title('Recording_1')
plt.plot(timing, signal_1, c="#3ABFE7")
plt.ylim(-35000, 35000)
plt.show()

mix_2_wave = wave.open('/home/wasi/ML_FOLDER/DSND_Term1-master/lessons/Unsupervised/5_ICA/ICA mix 2.wav', 'r')

print(mix_2_wave.getparams())

signal_2_raw = mix_2_wave.readframes(-1)
# print(signal_1_raw)
signal_2 = np.fromstring(signal_2_raw, 'Int16')
# print(signal_1)
'length: ', len(signal_2), 'first 100 elements: ', signal_2[:100]

fs = mix_2_wave.getframerate()
timing = np.linspace(0, len(signal_2)/fs, num=len(signal_2))

plt.figure(figsize=(12, 22))
plt.title('Recording_2')
plt.plot(timing, signal_2, c="#3ABFE7")
plt.ylim(-35000, 35000)
plt.show()

mix_3_wave = wave.open('/home/wasi/ML_FOLDER/DSND_Term1-master/lessons/Unsupervised/5_ICA/ICA mix 3.wav', 'r')

print(mix_3_wave.getparams())

signal_3_raw = mix_3_wave.readframes(-1)
# print(signal_1_raw)
signal_3 = np.fromstring(signal_3_raw, 'Int16')
# print(signal_1)
'length: ', len(signal_3), 'first 100 elements: ', signal_3[:100]

fs = mix_3_wave.getframerate()
timing = np.linspace(0, len(signal_3)/fs, num=len(signal_3))

plt.figure(figsize=(12, 22))
plt.title('Recording_3')
plt.plot(timing, signal_3, c="#3ABFE7")
plt.ylim(-35000, 35000)
plt.show()

X = list(zip(signal_1, signal_2, signal_3))
print(X[:10])

ica = FastICA(n_components=3)
ica_result = ica.fit_transform(X)
print(ica_result.shape)

result_signal_1 = ica_result[:, 0]
result_signal_2 = ica_result[:, 1]
result_signal_3 = ica_result[:, 2]

plt.figure(figsize=(12, 2))
plt.title('Independent Component #1')
plt.plot(result_signal_1, c="#df8efd")
plt.ylim(-0.010, 0.010)
plt.show()

plt.figure(figsize=(12, 2))
plt.title('Independent Component #2')
plt.plot(result_signal_2, c="#df8efd")
plt.ylim(-0.010, 0.010)
plt.show()

plt.figure(figsize=(12, 2))
plt.title('Independent Component #2')
plt.plot(result_signal_3, c="#df8efd")
plt.ylim(-0.010, 0.010)
plt.show()

result_signal_1_int = np.int16(result_signal_1 * 32767 * 100)
result_signal_2_int = np.int16(result_signal_2 * 32767 * 100)
result_signal_3_int = np.int16(result_signal_3 * 32767 * 100)


wavfile.write("/home/wasi/ML_FOLDER/DSND_Term1-master/lessons/Unsupervised/5_ICA/result_signal_1.wav", fs,
              result_signal_1_int)
wavfile.write("/home/wasi/ML_FOLDER/DSND_Term1-master/lessons/Unsupervised/5_ICA/result_signal_2.wav", fs,
              result_signal_2_int)
wavfile.write("/home/wasi/ML_FOLDER/DSND_Term1-master/lessons/Unsupervised/5_ICA/result_signal_3.wav", fs,
              result_signal_3_int)





