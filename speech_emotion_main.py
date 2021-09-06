import os
import numpy as np
import matplotlib.pyplot as plt
import librosa as lr
import speech_recognition as sr
from scipy.io import wavfile as wav
from scipy import signal
from pasta.augment import inline
from scipy.fftpack import fft
from python_speech_features import mfcc,logfbank
from tqdm import tqdm
import glob
import pandas as pd
import soundfile
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
import pyaudio
import wave



#path of Ravdess Audio file
os.listdir(path='/Users/dipanshuraj/Downloads/Audio_Speech_Actors_01-24')
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

dirName = '/Users/dipanshuraj/Downloads/Audio_Speech_Actors_01-24'
listOfFiles = getListOfFiles(dirName)
#len(listOfFiles)
print (len(listOfFiles))


r=sr.Recognizer()
for file in range(0 , len(listOfFiles) , 1):
    with sr.AudioFile(listOfFiles[file]) as source:
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print(text)
        except: print('error')






#Plotting the Basic Graphs for understanding of Audio Files :

for file in range(0, len(listOfFiles), 1):
    audio, sfreq = lr.load(listOfFiles[file])
    time = np.arange(0, len(audio)) / sfreq

    fig, ax = plt.subplots()
    ax.plot(time, audio)
    ax.set(xlabel='Time (s)', ylabel='Sound Amplitude')
    plt.show()

#PLOT THE SEPCTOGRAM

for file in range(0 , len(listOfFiles) , 1):
     sample_rate , samples = wav.read(listOfFiles[file])
     frequencies , times, spectrogram = signal.spectrogram(samples, sample_rate)
     plt.pcolormesh(times, frequencies, spectrogram)
     plt.imshow(spectrogram)
     plt.ylabel('Frequency [Hz]')
     plt.xlabel('Time [sec]')
     plt.show()



#Now Cleaning Step is Performed where:
#DOWN SAMPLING OF AUDIO FILES IS DONE  AND PUT MASK OVER IT AND DIRECT INTO CLEAN FOLDER
#MASK IS TO REMOVE UNNECESSARY EMPTY VOIVES AROUND THE MAIN AUDIO VOICE
def envelope(y , rate, threshold):
    mask=[]
    y=pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate/10) ,  min_periods=1 , center = True).mean()
    for mean in y_mean:
        if mean>threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


# Next Step is In-Depth Visualisation of Audio Fiels and its certain features to plot for.
# They are the Plotting Functions to be called later.


print("cleaning")

for file in tqdm(glob.glob(r'/Users/dipanshuraj/Downloads/Audio_Speech_Actors_01-24/Actor_*/*.wav')):
    file_name = os.path.basename(file)
    signal , rate = lr.load(file, sr=16000)
    mask = envelope(signal,rate, 0.0005)
    wav.write(filename= r'/Users/dipanshuraj/Downloads/clean_data/'+str(file_name), rate=rate,data=signal[mask])



print("cleaning complete")




#Feature Extraction of Audio Files Function
#Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(lr.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(lr.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(lr.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(lr.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
    return result

#Emotions in the RAVDESS dataset to be classified Audio Files based on .
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
#These are the emotions User wants to observe more :
observed_emotions=['calm', 'happy', 'fearful', 'disgust']



#Load the data and extract features for each sound file
from glob import glob
import os
import glob
def load_data(test_size=0.33):
    x,y=[],[]
    answer = 0
    for file in glob.glob(r'/Users/dipanshuraj/Downloads/clean_data/*.wav'):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            answer += 1
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append([emotion,file_name])
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)








#Split the dataset


x_train,x_test,y_trai,y_tes=load_data(test_size=0.25)
print(np.shape(x_train),np.shape(x_test), np.shape(y_trai),np.shape(y_tes))
y_test_map = np.array(y_tes).T
y_test = y_test_map[0]
test_filename = y_test_map[1]
y_train_map = np.array(y_trai).T
y_train = y_train_map[0]
train_filename = y_train_map[1]
print(np.shape(y_train),np.shape(y_test))
print(*test_filename,sep="\n")



#Get the shape of the training and testing datasets
# print((x_train.shape[0], x_test.shape[0]))
print((x_train[0], x_test[0]))
#Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

# Initialize the Multi Layer Perceptron Classifier
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)


#Train the model
model.fit(x_train,y_train)


#SAVING THE MODEL

# Save the Modle to file in the current working directory
#For any new testing data other than the data in dataset

Pkl_Filename = "Emotion_Voice_Detection_Model.pkl"

with open(Pkl_Filename, 'wb') as file:
    pickle.dump(model, file)



# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:
    Emotion_Voice_Detection_Model = pickle.load(file)

Emotion_Voice_Detection_Model



#predicting :
y_pred=Emotion_Voice_Detection_Model.predict(x_test)

print(y_pred)


#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


#Store the Prediction probabilities into CSV file

y_pred1 = pd.DataFrame(y_pred, columns=['predictions'])
y_pred1['file_names'] = test_filename
print(y_pred1)
y_pred1.to_csv('predictionfinal.csv')

'''
'''
# import the time module
import time


# define the countdown func.
def countdown(t):
    while t:
        mins, secs = divmod(t, 60)

        timer = '{:02d}:{:02d}'.format(mins, secs)
        print("start recording in",timer)
        time.sleep(1)
        t -= 1

    print('Fire in the hole!!')


# input time in seconds


# function call
countdown(int(3))


#RECORDED USING MICROPHONE:
CHUNK = 1024
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 1
RATE = 44100 #sample rate
RECORD_SECONDS = 4
WAVE_OUTPUT_FILENAME = "output10.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

print("* recording")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


#The file 'output10.wav' in the next cell is the file that was recorded live using the code :
data, sampling_rate = lr.load('output10.wav')

import librosa.display

plt.figure(figsize=(15, 5))
lr.display.waveplot(data, sr=sampling_rate)



## Appying extract_feature function on random file and then loading model to predict the result
file = '/Users/dipanshuraj/PycharmProjects/guvi_projects/venv/output10.wav'
# data , sr = librosa.load(file)
# data = np.array(data)
ans =[]
new_feature  = extract_feature(file, mfcc=True, chroma=True, mel=True)
ans.append(new_feature)
ans = np.array(ans)
# data.shape

rand_prid=Emotion_Voice_Detection_Model.predict(ans)
print(rand_prid)