
import torch
import torch.nn as nn
from mir_eval.separation import bss_eval_sources
import numpy as np


def tensor_normalize(S):
    temp_max, _ = torch.max(S, dim=1)
    batch_max, _ = torch.max(temp_max, dim=1)
    batch_max = torch.reshape(batch_max, (1, 1, 1))
    temp_min, _ = torch.min(S, dim=1)
    batch_min, _ = torch.min(temp_min, dim=1)
    batch_min = torch.reshape(batch_min, (1, 1, 1))
    normalized_S = (S - batch_min) / (batch_max - batch_min)
    return normalized_S


def validate_hide(audio, model, embedder, testloader, writer, step):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in testloader:
            dvec_mel, eliminated_wav, mixed_wav, expected_hidden_wav, eliminated_mag, expected_hidden_mag, mixed_mag, mixed_phase, _, _, _ = batch[0]

            dvec_mel = dvec_mel.cuda()
            eliminated_mag = eliminated_mag.unsqueeze(0).cuda()
            mixed_mag = mixed_mag.unsqueeze(0).cuda()
            expected_hidden_mag = expected_hidden_mag.unsqueeze(0).cuda()
            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)
            shadow_mag = model(mixed_mag, dvec)
            # shadow_mag.size() = [1, 301, 601]
            recorded_mag = tensor_normalize(mixed_mag + shadow_mag)
            test_loss = criterion(expected_hidden_mag, recorded_mag).item()

            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            eliminated_mag = eliminated_mag[0].cpu().detach().numpy()
            expected_hidden_mag = expected_hidden_mag[0].cpu().detach().numpy()
            recorded_mag = recorded_mag[0].cpu().detach().numpy()
            shadow_mag = shadow_mag[0].cpu().detach().numpy()
            shadow_wav = audio.spec2wav(shadow_mag, mixed_phase)
            scale = np.max(mixed_mag + shadow_mag) - np.min(mixed_mag + shadow_mag)
            # scale is frequency pass to time domain, used on wav signal normalization
            recorded_wav1 = audio.spec2wav(recorded_mag, mixed_phase)  # path 1
            recorded_wav2 = (mixed_wav + 50*shadow_wav) / max(abs(mixed_wav + 50*shadow_wav))  # path 2
            recorded_wav3 = (mixed_wav + shadow_wav) / max(abs(mixed_wav + shadow_wav))  # path 3
            sdr = bss_eval_sources(expected_hidden_wav, recorded_wav1, False)[0][0]
            # do normalize wav or not?
            writer.log_evaluation_hide(test_loss, sdr,
                                  mixed_wav, shadow_wav, recorded_wav1, recorded_wav2, recorded_wav3,
                                  eliminated_wav, expected_hidden_wav,
                                  mixed_mag.T, eliminated_mag.T, expected_hidden_mag.T, recorded_mag.T, shadow_mag.T,
                                  step)
            break
    model.train()


def validate_focus(audio, model, embedder, testloader, writer, step):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in testloader:
            # ref_mel, expected_focused_wav, mixed_wav, expected_focused_mag, mixed_mag, mixed_phase, dvec_path, expected_focused_wav_path, mixed_wav_path
            dvec_mel, expected_focused_wav, mixed_wav, expected_focused_mag, mixed_mag, mixed_phase, _, _, _ = batch[0]
            print(batch[0][-1])
            dvec_mel = dvec_mel.cuda()
            expected_focused_mag = expected_focused_mag.unsqueeze(0).cuda()
            mixed_mag = mixed_mag.unsqueeze(0).cuda()

            dvec = embedder(dvec_mel)
            dvec = dvec.unsqueeze(0)
            shadow_mag = model(mixed_mag, dvec)
            # shadow_mag.size() = [1, 301, 601]
            recorded_mag = tensor_normalize(mixed_mag + shadow_mag)
            test_loss = criterion(expected_focused_mag, recorded_mag).item()

            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            expected_focused_mag = expected_focused_mag[0].cpu().detach().numpy()
            recorded_mag = recorded_mag[0].cpu().detach().numpy()
            shadow_mag = shadow_mag[0].cpu().detach().numpy()
            shadow_wav = audio.spec2wav(shadow_mag, mixed_phase)
            scale = np.max(mixed_mag + shadow_mag) - np.min(mixed_mag + shadow_mag)
            # scale is frequency pass to time domain, used on wav signal normalization
            recorded_wav1 = audio.spec2wav(recorded_mag, mixed_phase)  # path 1
            recorded_wav2 = (mixed_wav + 50 * shadow_wav) / max(abs(mixed_wav + 50 * shadow_wav))  # path 2
            recorded_wav3 = (mixed_wav + shadow_wav) / max(abs(mixed_wav + shadow_wav))  # path 3
            sdr = bss_eval_sources(expected_focused_wav, recorded_wav1, False)[0][0]
            writer.log_evaluation_focus(test_loss, sdr,
                                  mixed_wav, shadow_wav, recorded_wav1, recorded_wav2, recorded_wav3,
                                  expected_focused_wav,
                                  mixed_mag.T, expected_focused_mag.T, recorded_mag.T, shadow_mag.T,
                                  step)
            break

    model.train()
