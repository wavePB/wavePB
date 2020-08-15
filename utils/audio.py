# adapted from Keith Ito's tacotron implementation
# https://github.com/keithito/tacotron/blob/master/util/audio.py

import librosa
import numpy as np
import torch


class Audio():
    def __init__(self, hp):
        self.hp = hp
        self.mel_basis = librosa.filters.mel(sr=hp.audio.sample_rate,
                                             n_fft=hp.embedder.n_fft,
                                             n_mels=hp.embedder.num_mels)

    def get_mel(self, y):
        y = librosa.core.stft(y=y, n_fft=self.hp.embedder.n_fft,
                              hop_length=self.hp.audio.hop_length,
                              win_length=self.hp.audio.win_length,
                              window='hann')
        magnitudes = np.abs(y) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel

    def wav2spec(self, y):
        D = self.stft(y)
        S = self.amp_to_db(np.abs(D)) - self.hp.audio.ref_level_db
        S, D = self.normalize(S), np.angle(D)
        S, D = S.T, D.T  # to make [time, freq]
        return S, D

    def tensorwav2spec(self, wav):
        D = self.tensorstft(wav)  # D.size() = (601, 301, 2)
        absD = torch.sqrt(torch.square(D[:, :, 0]) + torch.square(D[:, :, 1]))
        S = self.tensoramp_to_db(absD) - self.hp.audio.ref_level_db
        S = self.tensornormalize(S)
        S = torch.transpose(S, 0, 1)
        return S

    def batchwav2spec(self, wavs):
        """
        This function takes batch size of wav Tensor, and return a batch of spectrogram tensor
        Args:
            wavs: batch of audios tensor

        Returns: batch of spectrograms tensor
        """
        spectrograms = list()
        for wav in wavs:
            S = self.tensorwav2spec(wav)
            spectrograms.append(S)
        spectrograms = torch.stack(spectrograms, dim=0)
        return spectrograms

    def spec2wav(self, spectrogram, phase):
        spectrogram, phase = spectrogram.T, phase.T
        # used during inference only
        # spectrogram: enhanced output
        # phase: use noisy input's phase, so no GLA is required
        S = self.db_to_amp(self.denormalize(spectrogram) + self.hp.audio.ref_level_db)
        return self.istft(S, phase)

    def tensorspec2wav(self, spectrogram, phase):
        spectrogram = torch.transpose(spectrogram, 0, 1)
        phase = torch.transpose(phase, 0, 1)
        # spectrogram, phase = spectrogram.T, phase.T
        S = self.tensordb_to_amp(self.tensordenormalize(spectrogram) + self.hp.audio.ref_level_db)
        # wav = self.istft(S, phase)
        wav = self.tensoristft(S, phase)
        return wav

    def batchspec2wav(self, spectrograms, phases):
        """
        This function takes batch size of spectrogram and phase Tensor, and return a batch of wav tensor
        Args:
            spectrograms: batch of spectrogram. (batch_size, nfft/2, time*sp/hop_len)
            phases: batch of phases. (batch_size, nfft/2, time*sp/hop_len)

        Returns:
            time domain signal reconstructed from stft_matrix
        """
        wav_list = list()
        for spectrogram, phase in zip(spectrograms, phases):
            wav = self.tensorspec2wav(spectrogram, phase)
            wav_list.append(wav)
        wav_list = torch.stack(wav_list, dim=0)
        # print(wav_list.shape)   # [batch_size, 48000]
        return wav_list

    def stft(self, y):
        return librosa.stft(y=y, n_fft=self.hp.audio.n_fft,
                            hop_length=self.hp.audio.hop_length,
                            win_length=self.hp.audio.win_length)

    def tensorstft(self, y):
        return torch.stft(input=y, n_fft=self.hp.audio.n_fft,
                          hop_length=self.hp.audio.hop_length,
                          win_length=self.hp.audio.win_length)

    def istft(self, mag, phase):
        stft_matrix = mag * np.exp(1j * phase)
        return librosa.istft(stft_matrix,
                             hop_length=self.hp.audio.hop_length,
                             win_length=self.hp.audio.win_length)

    def tensoristft(self, mag, phase):
        stft_real = mag * torch.cos(phase)
        stft_imag = mag * torch.sin(phase)
        istftin = torch.stack((stft_real, stft_imag), dim=2)
        signal = torch.istft(istftin, n_fft=self.hp.audio.n_fft, hop_length=self.hp.audio.hop_length,
                             win_length=self.hp.audio.win_length)
        return signal

    def amp_to_db(self, x):
        return 20.0 * np.log10(np.maximum(1e-5, x))

    def tensoramp_to_db(self, x):
        return 20.0 * torch.log10(torch.clamp(x, min=1e-5))

    def db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def tensordb_to_amp(self, x):
        return torch.pow(10.0, x * 0.05)

    def normalize(self, S):
        return np.clip(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def tensornormalize(self, S):
        return torch.clamp(S / -self.hp.audio.min_level_db, -1.0, 0.0) + 1.0

    def denormalize(self, S):
        return (np.clip(S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db

    def tensordenormalize(self, S):
        return (torch.clamp(S, 0.0, 1.0) - 1.0) * -self.hp.audio.min_level_db
