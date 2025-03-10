import os
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torchaudio
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import random
import torchaudio.transforms as T
from scipy.fftpack import idct
from tqdm import tqdm

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

def load_waveform(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    
    if waveform.shape[0] > 1:
        waveform = waveform[0, :]
    
    return waveform, sample_rate

def extract_features(waveform, sample_rate, n_mfcc=13, frame_length=2048, hop_length=512):
    num_frames = (waveform.numel() - frame_length) // (hop_length + 1)
    frames = waveform.unfold(0, frame_length, hop_length)

    signs = frames.sign()
    zcr = ((signs[:, :-1] * signs[:, 1:]) < 0).float().sum(dim=1) / frame_length

    ste = (frames ** 2).sum(dim=1) / frame_length
    
    # TODO: MFCC from scratch
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate, 
        n_mfcc=n_mfcc, 
        melkwargs={
            'n_fft': frame_length,
            'hop_length': hop_length,
            'n_mels': 40
        }
    )
    mfcc = mfcc_transform(waveform.unsqueeze(0))
    
    return zcr, ste, mfcc

def zcr_plot_violin(zcr1, zcr2, sp1, sp2) -> None:
    dat = pd.DataFrame({
        'zcr' : torch.cat([zcr1, zcr2]).numpy(),
        "Speaker": [sp1] * len(zcr1) + [sp2] * len(zcr2)
    })
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Speaker", y="zcr", data=dat, inner="box", palette="Set2")
    plt.title("Zero-Crossing Rate Distribution Comparison")
    plt.xlabel("Speaker")
    plt.ylabel("Zero-Crossing Rate")
    plt.show()

def zcr_plot(zcr1, zcr2, sp1, sp2) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(zcr1, label=sp1)
    plt.plot(zcr2, label=sp2)
    plt.title("Zero-Crossing Rate Comparison")
    plt.xlabel("Frame")
    plt.ylabel("Zero-Crossing Rate")
    plt.legend()
    plt.show()
    
def ste_plot(ste1, ste2, sp1, sp2) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(ste1.numpy(), label=sp1, marker='o', linestyle='', alpha=0.8)
    plt.plot(ste2.numpy(), label=sp2, marker='o', linestyle='', alpha=0.8)
    plt.title("Short-Time Energy Over Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Short-Time Energy")
    plt.legend()
    plt.show()
    
def ste_plot_violin(ste1, ste2, sp1, sp2) -> None:
    dat = pd.DataFrame({
        'ste' : torch.cat([ste1, ste2]).numpy(),
        "Speaker": [sp1] * len(ste1) + [sp2] * len(ste2)
    })
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(x="Speaker", y="ste", data=dat, inner="box", palette="Set2")
    plt.title("Short-Time Energy Distribution Comparison")
    plt.xlabel("Speaker")
    plt.ylabel("Short-Time Energy")
    plt.show()

def mfcc_plot(mfcc, sp) -> None:
    plt.figure(figsize=(10, 6))
    plt.imshow(mfcc[0].squeeze(0).numpy(), aspect='auto', cmap='viridis')
    plt.title("MFCC: " + sp)
    plt.xlabel("Frame")
    plt.ylabel("MFCC Coefficient")
    plt.colorbar(label="Coefficient Value")
    plt.show()    

def compute_spectral_envelopes(mfcc_tensor, n_mels=40):
    mfcc_np = mfcc_tensor.numpy()
    T_frames = mfcc_np.shape[1]
    envelopes = []
    for t in range(T_frames):
        log_mel_env = idct(mfcc_np[:, t], n=n_mels, norm='ortho')
        envelope = np.exp(log_mel_env)
        envelopes.append(envelope)
    envelopes = np.array(envelopes)
    return envelopes

def plot_speaker_envolopes(mfcc1, mfcc2, sp1, sp2) -> None:
    mfcc1 = mfcc1[0]
    mfcc2 = mfcc2[0]
    
    env1 = compute_spectral_envelopes(mfcc1)
    env2 = compute_spectral_envelopes(mfcc2)
    
    mean_env1 = env1.mean(axis=0)
    std_env1 = env1.std(axis=0)
    
    mean_env2 = env2.mean(axis=0)
    std_env2 = env2.std(axis=0)
    
    mel_bins = np.arange(mean_env1.shape[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(mel_bins, mean_env1, label=sp1, color='blue')
    plt.fill_between(mel_bins, mean_env1 - std_env1, mean_env1 + std_env1, color='blue', alpha=0.3)
    plt.plot(mel_bins, mean_env2, label=sp2, color='red')
    plt.fill_between(mel_bins, mean_env2 - std_env2, mean_env2 + std_env2, color='red', alpha=0.3)
    plt.xlabel("Mel Frequency Bin")
    plt.ylabel("Amplitude")
    plt.title("Average Spectral Envelope Reconstructed from MFCCs")
    plt.legend()
    plt.show()
if __name__ == '__main__':
    pass
