
import os
import pickle
import librosa
import numpy as np    

model, encoder, scaler = None, None, None

def get_model():
    print('\n\n\n\n\n\n\n\n')
    from keras.models import Sequential
    from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
    model=Sequential()
    model.add(Conv1D(256*2, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(162, 1)))
    model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    model.add(Conv1D(256*2, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    model.add(Conv1D(128*2, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))
    model.add(Dropout(0.2))

    model.add(Conv1D(64*2, kernel_size=5, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=5, strides = 2, padding = 'same'))

    model.add(Flatten())
    model.add(Dense(units=32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(units=8, activation='softmax'))
    model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

# model.summary()

frida_emotions = ['amusement', 'awe', 'contentment', 'excitement',
        'anger', 'disgust', 'fear', 'sadness', 'something else']

# Need to map the emotions from the speech2emotion datasets to FRIDA's emotions
emotion_map = {
    'angry':'anger',
    'fear':'fear',
    'calm':'contentment',
    'happy':'amusement',
    'sad':'sadness',
    'neutral':'something else',
    'disgust':'disgust',
    'surprise':'excitement'
}

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally

    return result

def to_features(path):
    data, sample_rate = librosa.load(path)#, duration=2.5, offset=0.6)

    feat = extract_features(data, sample_rate)
    feat = np.array(feat)[None,:]
    # print(feat.shape, feat.mean(), feat.std())
    feat = scaler.transform(feat)
    # print(feat.shape, feat.mean(), feat.std())
    feat = np.expand_dims(feat, axis=2)
    return feat


def speech2emotion(path):
    # Path is a path to a wav file
    global model, encoder, scaler
    if model is None:
        model = get_model()
    root = os.path.dirname(os.path.realpath(__file__))


    model.load_weights(os.path.join(root, "speech2emotion_weights.h5"))

    with open(os.path.join(root, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(root, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    feat = to_features(path)
    pred = model.predict(feat)
    # print(pred)
    cat = encoder.inverse_transform(pred)[0][0]
    print('Predicted Emotion', cat)
    frida_cat = emotion_map[cat]
    emotion_vec = np.zeros(len(frida_emotions))
    emotion_vec[frida_emotions.index(frida_cat)] = 1.0
    # print(emotion_vec)
    return emotion_vec

def speech2text(path):
    import whisper

    model = whisper.load_model("base")
    result = model.transcribe(path)
    print('Transcribed text', result["text"])
    return result["text"]

if __name__ == '__main__':
    root = os.path.dirname(os.path.realpath(__file__))
    speech2emotion(os.path.join(root, 'speech_example.wav'))
    text = speech2text(os.path.join(root, 'speech_example.wav'))
    print(text)
