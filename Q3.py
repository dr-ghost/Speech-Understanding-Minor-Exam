import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
import torchaudio.transforms as T
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import itertools

def rectangular_window(audio : torch.Tensor, window_size : int, num_windows : int):
    """
    audio: (num_channels, num_frames)
    """
    num_channels, num_frames = audio.shape
    
    windows = torch.zeros(num_channels, num_windows, window_size)
    
    hop_size = (num_frames - window_size) //(num_windows - 1)
    
    for i in range(num_windows - 1):
        windows[:, i] = audio[:, i*hop_size:i*hop_size+window_size]
    
    windows[:, num_windows - 1] = audio[:, -window_size:]
    
    return windows

def hamming_window(audio : torch.Tensor, window_size : int, num_windows : int):
    windows = rectangular_window(audio, window_size, num_windows)
    
    hamming_win = torch.signal.windows.hamming(window_size)
    
    return windows * hamming_win

def hanning_window(audio : torch.Tensor, window_size : int, num_windows : int):
    windows = rectangular_window(audio, window_size, num_windows)
    
    hamming_win = torch.signal.windows.hann(window_size)
    
    return windows * hamming_win

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    
def plot_spectrogram(file_path, genre_title, n_fft=1024, hop_length=None):
    audio, sample_rate = torchaudio.load(file_path)
    
    if audio.shape[0] > 1:
        audio = audio[0, :]
        
    audio = audio[:4_00_000]

    spectrogram_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    
    spec = spectrogram_transform(audio)
        
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    spec_db = db_transform(spec)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.numpy(), origin='lower', aspect='auto', 
               extent=[0, spec_db.shape[1] * (hop_length or (n_fft // 2)) / sample_rate, 0, sample_rate / 2])
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram - {genre_title}")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
    
def plot_spectrogram_from_scratch(file_path, genre_title, n_fft=1024, hop_length=None):
    audio, sample_rate = torchaudio.load(file_path)
    
    # if audio.shape[0] > 1:
    #     audio = audio[0, :]
        
    audio = audio[:4_00_000]

    spec = torch.abs(torch.fft.rfft(hanning_window(audio, 1_000, 400), dim=-1))
    
    if spec.shape[0] > 1:
        spec = spec[0, ...]
    
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    spec_db = db_transform(spec)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.numpy(), origin='lower', aspect='auto', 
               extent=[0, spec_db.shape[1] * (hop_length or (n_fft // 2)) / sample_rate, 0, sample_rate / 2])
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram - {genre_title}")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
    
def plot_spectorgram_mediate(features):
    
    spec = torch.abs(torch.fft.rfft(features, dim=-1))
    
    if spec.shape[0] > 1:
        spec = spec[0, ...]
    
    db_transform = T.AmplitudeToDB(stype='power', top_db=80)
    spec_db = db_transform(spec)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec_db.numpy(), origin='lower', aspect='auto', 
               extent=[0, spec_db.shape[1] * 320 / 8000, 0, 8000 / 2])
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram")
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()
    
def levinson_durbin(r, order):
    a = np.zeros(order + 1)
    e = r[0]
    if e == 0:
        return a, e
    a[0] = 1.0
    for i in range(1, order + 1):
        acc = 0.0
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k = (r[i] - acc) / e
        a[i] = k
        for j in range(1, i):
            a[j] = a[j] - k * a[i - j]
        e *= (1 - k**2)
    return a, e

def compute_lpc(signal, order):
    sign = signal.squeeze().numpy()
    #print(sign.shape)
    auto_crr = np.correlate(sign, sign, mode='full')
    mid = auto_crr.shape[-1] // 2
    r = auto_crr[mid:mid + order + 1]
    a, e = levinson_durbin(r, order)
    return a

def extract_formants(signal, sr, lpc_order=12):
    a = compute_lpc(signal, lpc_order)

    roots = np.roots(a)
    
    roots = [root for root in roots if np.imag(root) >= 0.005]
    
    angles = np.angle(roots)
    
    freqs = sorted(angles * (sr / (2 * np.pi)))
    
    
    if len(freqs) < 3:
        freqs += [0]*(3 - len(freqs))
    return freqs[0], freqs[1], freqs[2]

def extract_f0(signal, sr):
    sign = signal.squeeze().numpy()
    
    sign = sign - np.mean(sign)
    
    autocorr = np.correlate(sign, sign, mode='full')
    
    autocorr = autocorr[len(autocorr)//2:]
    
    min_lag = sr // 500
    max_lag = sr // 50
    if max_lag >= autocorr.shape[-1]:
        max_lag = autocorr.shape[-1] - 1
    
    peak_index = np.argmax(autocorr[min_lag:max_lag]) + min_lag
    if peak_index == 0:
        return 0
    f0 = sr / peak_index
    return f0

def extract_fundamental_features(signal, sr, spc_order=12):
    """
    signal: (batch, frame, ...)
    """
    feature_vec = torch.zeros(signal.shape[0], signal.shape[1], 4)
    for j in range(signal.shape[0]):
        for i in range(signal.shape[1]):
            f0 = extract_f0(signal[j, i, :], sr)
            f1, f2, f3 = extract_formants(signal[j, i, :], sr, spc_order)
            
            feature_vec[j, i, :] = torch.tensor([f0, f1, f2, f3])
    
    return torch.flatten(feature_vec, start_dim=1)

def load_audio(file_path):
    audio, sr = torchaudio.load(file_path)
    return audio, sr

class VowelDataset(Dataset):
    def __init__(self, base_path, window_function):
        self.base_path = base_path
        self.file_path_lst = []
        
        for root, sub_dir, files in os.walk(base_path):
            for file in files:
                if file.endswith(".wav"):
                    self.file_path_lst.append(os.path.join(root, file)) 
           
        if window_function is None:     
            pass
        elif window_function.__name__ == "rectangular_window":
            self.num_windows = 50
            self.window_size = 180
        else:
            self.num_windows = 50
            self.window_size = 210
        
        self.window_function = window_function
        
        self.vowels = {vow:i for i, vow in enumerate(['a', 'e', 'i', 'o', 'u'])}
    
    def __len__(self):
        return len(self.file_path_lst)
    
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.file_path_lst[idx])
        
        #waveform = torchaudio.functional.resample(waveform, 44100, 8000)
        
        if waveform.shape[0] > 1:
            waveform = waveform[0, :]
        
        label = self.vowels[self.file_path_lst[idx].split("/")[-2][-1]]
        
        if self.window_function is not None:    
            windowed_features = self.window_function(waveform, self.window_size, self.num_windows)
        else:
            windowed_features = waveform

        return waveform, windowed_features, sr, label
    
    
class KNNVowelClassifier:
    def __init__(self, k):
        self.k = k
    
    def train(self, X_train, y_train = None) -> None:
        if y_train is None:
            self.X_train = torch.zeros(len(X_train), X_train[0][1].shape[1]*4)
            self.y_train = torch.zeros(len(X_train))
            
            for i, (waveform, features, sr, label) in enumerate(X_train):
                self.X_train[i] = extract_fundamental_features(features, sr)
                self.y_train[i] = label
        else:
            self.X_train = X_train
            self.y_train = y_train
        
    def predict(self, X):
        dist = torch.cdist(X, self.X_train)
        
        indxs = dist.topk(self.k, largest=False).indices
        
        trained_labls = self.y_train[indxs]
        
        preds = torch.mode(trained_labls, dim=1).values
        
        return preds
    
    def evaluate(self, X_test, y_test = None):
        if y_test is None:
            sX_test = torch.zeros(len(X_test), X_test[0][1].shape[1]*4)
            sy_test = torch.zeros(len(X_test))
            
            for i, (waveform, features, sr, label) in enumerate(X_test):
                sX_test[i] = extract_fundamental_features(features, sr)
                sy_test[i] = label
        else:
            sX_test = X_test
            sy_test = y_test
        
        preds = self.predict(sX_test)
        
        return accuracy_score(sy_test, preds), f1_score(sy_test, preds, average='macro'), confusion_matrix(sy_test, preds)
    
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap='Blues'):
    if normalize:
        cm = np.where(cm.sum(axis=1, keepdims=True) == 0, 0, cm.astype('float') / cm.sum(axis=1, keepdims=True))
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap,
                xticklabels=classes, yticklabels=classes, cbar=True, linewidths=0.5)
    
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
