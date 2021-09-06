import os

from scipy.io import wavfile as wav

import matplotlib.pyplot as plt
import librosa.display


os.listdir(path='/Users/dipanshuraj/Desktop/imagine audio')
def getListOfFiles(dirName):
    listOfFile=os.listdir(dirName)
    allFiles=list()
    for entry in listOfFile:
        fullPath=os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles=allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
    return allFiles

dirName = '/Users/dipanshuraj/Desktop/imagine audio'
listOfFiles = getListOfFiles(dirName)
#len(listOfFiles)
print (len(listOfFiles))


import librosa
audio_path = '/Users/dipanshuraj/Desktop/imagine audio/03-01-06-02-01-01-01.wav'
x , sr = librosa.load(audio_path)
samples = wav.read(audio_path)
sample_rate=16000

#display waveform



plt.figure(figsize=(14, 5))

librosa.display.waveplot(x, sr=sample_rate)


#display Spectrogram

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
#If to pring log of frequencies
#librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='log')
plt.colorbar()
plt.show()




mfccs = librosa.feature.mfcc(x, sr=sample_rate)
print(mfccs.shape)
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
plt.show()
