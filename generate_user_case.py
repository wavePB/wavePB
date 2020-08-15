from datasets.eva_dataloader_focus import create_dataloader
import wavio
import argparse
from utils.hparams import HParam
import os
import glob
import torch
import librosa
import argparse

from utils.audio import Audio
from model.model import VoiceFilter
from model.embedder import SpeechEmbedder
from utils.evaluation import tensor_normalize


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--base_dir', type=str, default='.',
                        help="Root directory of run.")
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-e', '--embedder_path', type=str, required=True,
                        help="path of embedder model pt file")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file of focus model")
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="Name of the model. Used for both logging and saving checkpoints.")
    parser.add_argument('-g', '--gpu', type=int, required=True, default='1',
                        help="ID of the selected gpu. Used for gpu selection.")
    parser.add_argument('-o', '--out_dir', type=str, required=True,
                        help="out directory of result.wav")
    args = parser.parse_args()

    hp = HParam(args.config)
    root_dir_test = hp.data.test_dir
    alldirs = [x[0] for x in os.walk(root_dir_test)]
    dirs = [leaf for leaf in alldirs if len(leaf.split('/'))>5]
    speaker_count = 0

    # dir = '/data/our_dataset/test/3/joint'
    for dir in dirs:
        speaker_count = speaker_count + 1
        print("Speaker : {}/56\n".format(speaker_count))
        tree = dir.split('/')
        speaker_id = tree[-2]
        hp.data.test_dir = dir
        testloader = create_dataloader(hp, args, train=False)
        for batch in testloader:
            # length of batch is 1, set in dataloader
            ref_mel, expected_focused_wav, mixed_wav, expected_focused_mag, mixed_mag, mixed_phase, dvec_path, expected_focused_wav_path, mixed_wav_path = \
                batch[0]
            # print("expected_focused: {}".format(expected_focused_wav_path))
            print("Mixed: {}".format(mixed_wav_path))
            model = VoiceFilter(hp).cuda()
            chkpt_model = torch.load(args.checkpoint_path)['model']
            model.load_state_dict(chkpt_model)
            model.eval()

            embedder = SpeechEmbedder(hp).cuda()
            chkpt_embed = torch.load(args.embedder_path)
            embedder.load_state_dict(chkpt_embed)
            embedder.eval()

            audio = Audio(hp)
            dvec_wav, _ = librosa.load(dvec_path, sr=16000)
            ref_mel = audio.get_mel(dvec_wav)
            ref_mel = torch.from_numpy(ref_mel).float().cuda()
            dvec = embedder(ref_mel)
            dvec = dvec.unsqueeze(0)  # (1, 256)

            mixed_wav, _ = librosa.load(mixed_wav_path, sr=16000)
            mixed_mag, mixed_phase = audio.wav2spec(mixed_wav)
            mixed_mag = torch.from_numpy(mixed_mag).float().cuda()

            mixed_mag = mixed_mag.unsqueeze(0)

            shadow_mag = model(mixed_mag, dvec)
            # shadow_mag.size() = [1, 301, 601]

            recorded_mag = tensor_normalize(mixed_mag + shadow_mag)
            recorded_mag = recorded_mag[0].cpu().detach().numpy()
            mixed_mag = mixed_mag[0].cpu().detach().numpy()
            expected_focused_mag = expected_focused_mag[0].cpu().detach().numpy()
            # recorded_mag = recorded_mag[0].cpu().detach().numpy()
            shadow_mag = shadow_mag[0].cpu().detach().numpy()
            shadow_wav = audio.spec2wav(shadow_mag, mixed_phase)

            # scale is frequency pass to time domain, used on wav signal normalization
            recorded_wav1 = audio.spec2wav(recorded_mag, mixed_phase)  # path 1

            # mixed_Wav_path = '/data/our_dataset/test/13/babble/000001-mixed.wav'
            focused1 = mixed_wav_path[:-9] + 'focused1.wav'
            focused2 = mixed_wav_path[:-9] + 'focused2.wav'
            # purified3 = os.path.join(args.out_dir, 'result3.wav')

            # original mixed wav and expected_focused wav are not PCM, cannot be read by google cloud
            wavio.write(focused1, recorded_wav1, 16000, sampwidth=2)  # frequency +
            wavio.write(focused2, shadow_wav, 16000, sampwidth=2)  # est noise
            # wavio.write(purified3, enhanced_wav, 16000, sampwidth=2)  # mix + est noise
