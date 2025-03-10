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
import torchaudio.functional as F
from scipy.fftpack import idct
from tqdm import tqdm
from scipy.signal import periodogram

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

    waveform = np.abs(waveform)

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
    figure.suptitle("Amplitude")

def plot_spectogram(signal: torch.Tensor, sample_rate: int, nfft: int = 1024, noverlap: int = 512):
    signal = signal.squeeze().numpy()
    
    plt.figure(figsize=(12, 6))

    spectrum, freqs, bins, im = plt.specgram(signal, Fs=sample_rate, NFFT=nfft, noverlap=noverlap, cmap='viridis')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Spectrogram')
    plt.colorbar(im, label='Intensity (dB)')
    plt.tight_layout()
    plt.show()    

def _plot_spectrogram(file_path, genre_title, n_fft=1024, hop_length=None):
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


def plot_pitch(waveform: torch.Tensor, sample_rate: int):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    
    pitch_transform = T.ComputeKaldiPitch(sample_rate=sample_rate)
    
    pitch_tensor = pitch_transform(waveform)
    
    pitch_values = pitch_tensor[:, 0, 0]  # shape: (num_frames,)
    
    frame_shift_sec = 10 / 1000.0  # 10 ms in seconds
    num_frames = pitch_values.shape[0]
    times = torch.arange(num_frames, dtype=torch.float32) * frame_shift_sec
    
    plt.figure(figsize=(12, 4))
    plt.plot(times.numpy(), pitch_values.numpy(), label="Pitch (Hz)", color="green")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title("Pitch Contour using torchaudio.transforms.ComputeKaldiPitch")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def plot_frequency_spectrum(signal: torch.Tensor, sample_rate: int):
    signal = signal.squeeze().numpy()
    
    frequencies, power_density = periodogram(signal, fs=sample_rate)
    
    plt.figure(figsize=(12, 4))
    plt.semilogy(frequencies, power_density, label='Power Spectral Density', color='red')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD')
    plt.title('Frequency Spectrum (Periodogram)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
def plot_rms_energy(waveform: torch.Tensor, sample_rate: int, frame_size: int = 1024, hop_length: int = 512):
    waveform = waveform.squeeze()
    
    frames = waveform.unfold(0, frame_size, hop_length)  # Shape: (num_frames, frame_size)
    
    rms_energy = torch.sqrt((frames ** 2).mean(dim=1))
    
    frame_times = torch.arange(rms_energy.size(0), dtype=torch.float32) * hop_length / sample_rate + (frame_size / (2 * sample_rate))
    
    plt.figure(figsize=(12, 4))
    plt.plot(frame_times.numpy(), rms_energy.numpy(), color='purple', label='RMS Energy')
    plt.xlabel('Time (s)')
    plt.ylabel('RMS Energy')
    plt.title('RMS Energy Over Time')
    plt.legend()
    plt.tight_layout()
    plt.show()