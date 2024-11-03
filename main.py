import numpy as np
import librosa
import os
from percepetron import Percepetron, LoadModel

PREDICTFILE = "nonclaps/gameover.wav"
LEARNINGRATE = 1

claps = os.listdir("claps")
nonclaps = os.listdir("nonclaps")
np.random.seed(30)

def extract_features(file_path) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(features, axis=1)

def load_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for file in claps:
        features = extract_features(f"claps/{file}")

        # Reshape features to 2D array for compatibility with Percepetron
        features = features
        X.append(features)
        y.append(1)
    

    
    for file in nonclaps:
        features = extract_features(f"nonclaps/{file}")
        features = features
        X.append(features)
        y.append(0)
    
    return np.array(X), np.array(y)

def main() -> None:
    X, y = load_data()
    percep = Percepetron(X, y.reshape(-1, 1), lr=0.1)
    percep.train(iterations=200000)
    accuracy = percep.accuracy(X, y.reshape(-1, 1))
    print(f"Acc: {accuracy: .2f}%")
    inputs = extract_features(PREDICTFILE)
    
    prediction = percep.predict(inputs)
    print(prediction)

def main2() -> None:
    #loading models
    model_path = "percep1.model"
    percep = LoadModel(model_path)
if __name__ == "__main__":
    main()


'''import numpy as np
import librosa
import os
from percepetron import Perceptron

PREDICTFILE = "claps/clap1.wav"
LEARNINGRATE = 12003
EPOCHS = 100
BATCH_SIZE = 32

claps = os.listdir("claps")
nonclaps = os.listdir("nonclaps")
np.random.seed(30)

def extract_features(file_path) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(features, axis=1)

def load_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for file in claps:
        features = extract_features(f"claps/{file}")
        X.append(features)
        y.append(1)
    
    for file in nonclaps:
        features = extract_features(f"nonclaps/{file}")
        X.append(features)
        y.append(0)
    
    return np.array(X), np.array(y)

def main() -> None:
    X, y = load_data()
    percep = Perceptron(X, y.reshape(-1, 1), lr=LEARNINGRATE)
    percep.train(epochs=EPOCHS, batch_size=BATCH_SIZE)
    accuracy = percep.accuracy(X, y.reshape(-1, 1))
    print(f"Final Accuracy: {accuracy:.2f}%")

    inputs = extract_features(PREDICTFILE)
    prediction = percep.predict(inputs)
    print(f"Prediction for {PREDICTFILE}: {prediction}")

if __name__ == "__main__":
    main()
'''