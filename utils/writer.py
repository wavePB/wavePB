import numpy as np
from tensorboardX import SummaryWriter

from .plotting import plot_spectrogram_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation_hide(self, test_loss, sdr,
                       mixed_wav, est_noise_wav, est_purified_wav1, est_purified_wav2, est_purified_wav3, eliminated_wav,
                       expected_hidden_wav,
                       mixed_spec, eliminated_spec, expected_hidden_spec, est_purified_mag, est_noise_mag,
                       step):
        self.add_scalar('test_loss', test_loss, step)
        self.add_scalar('SDR', sdr, step)

        self.add_audio('mixed_wav', mixed_wav, step, self.hp.audio.sample_rate)
        self.add_audio('eliminated_wav', eliminated_wav, step, self.hp.audio.sample_rate)
        self.add_audio('expected_hidden_wav', expected_hidden_wav, step, self.hp.audio.sample_rate)
        self.add_audio('est_noise_wav', est_noise_wav, step, self.hp.audio.sample_rate)
        self.add_audio('est_purified_wav1', est_purified_wav1, step, self.hp.audio.sample_rate)
        self.add_audio('est_purified_wav2', est_purified_wav2, step, self.hp.audio.sample_rate)
        self.add_audio('est_purified_wav3', est_purified_wav3, step, self.hp.audio.sample_rate)

        self.add_image('data/mixed_spectrogram',
                       plot_spectrogram_to_numpy(mixed_spec), step, dataformats='HWC')
        self.add_image('result/est_noise_spectrogram',
                       plot_spectrogram_to_numpy(est_noise_mag), step, dataformats='HWC')
        self.add_image('result/est_purified_spectrogram',
                       plot_spectrogram_to_numpy(est_purified_mag), step, dataformats='HWC')
        self.add_image('data/eliminated_spectrogram',
                       plot_spectrogram_to_numpy(eliminated_spec), step, dataformats='HWC')
        self.add_image('data/expected_hidden_spectrogram',
                       plot_spectrogram_to_numpy(expected_hidden_spec), step, dataformats='HWC')
        self.add_image('result/estimation_error_sq',
                       plot_spectrogram_to_numpy(np.square(est_purified_mag - expected_hidden_spec)), step,
                       dataformats='HWC')


    def log_evaluation_focus(self, test_loss, sdr,
                       mixed_wav, est_noise_wav, est_purified_wav1, est_purified_wav2, est_purified_wav3, expected_focusedwav,
                       mixed_spec, expected_focusedspec, est_purified_mag, est_noise_mag,
                       step):
        self.add_scalar('test_loss', test_loss, step)
        self.add_scalar('SDR', sdr, step)

        self.add_audio('mixed_wav', mixed_wav, step, self.hp.audio.sample_rate)
        self.add_audio('expected_focusedwav', expected_focusedwav, step, self.hp.audio.sample_rate)
        self.add_audio('est_noise_wav', est_noise_wav, step, self.hp.audio.sample_rate)
        self.add_audio('est_purified_wav1', est_purified_wav1, step, self.hp.audio.sample_rate)
        self.add_audio('est_purified_wav2', est_purified_wav2, step, self.hp.audio.sample_rate)
        self.add_audio('est_purified_wav3', est_purified_wav3, step, self.hp.audio.sample_rate)

        self.add_image('data/mixed_spectrogram',
                       plot_spectrogram_to_numpy(mixed_spec), step, dataformats='HWC')
        self.add_image('result/est_noise_spectrogram',
                       plot_spectrogram_to_numpy(est_noise_mag), step, dataformats='HWC')
        self.add_image('result/est_purified_spectrogram',
                       plot_spectrogram_to_numpy(est_purified_mag), step, dataformats='HWC')
        self.add_image('data/expected_focusedspectrogram',
                       plot_spectrogram_to_numpy(expected_focusedspec), step, dataformats='HWC')
        self.add_image('result/estimation_error_sq',
                       plot_spectrogram_to_numpy(np.square(est_purified_mag - expected_focusedspec)), step, dataformats='HWC')
