import matplotlib

matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np


def fig2np(fig):
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(5, 3))
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, 3, 0, 8000],
                   interpolation='none', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    plt.xlabel('Seconds')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()

    fig.canvas.draw()
    data = fig2np(fig)
    plt.close()
    return data


def plot_spectrogram(spectrogram, path):
    fig, ax = plt.subplots(figsize=(5, 3))
    spectrogram = spectrogram.T
    im = ax.imshow(spectrogram, aspect='auto', origin='lower', extent=[0, 3, 0, 2000],
                   interpolation='none', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    plt.xlabel('Seconds')
    plt.ylabel('Frequency (Hz)')
    plt.tight_layout()
    plt.savefig(path)
