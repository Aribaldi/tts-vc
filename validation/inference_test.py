import argparse
import json
import os
import string
import time
import numpy as np
import torch

from mozilla_TTS_utils.tts_generic_utils import setup_model
from mozilla_TTS_utils.synthesis import synthesis
from mozilla_TTS_utils.text.symbols import make_symbols
from mozilla_TTS_utils.audio import AudioProcessor
from mozilla_TTS_utils.io import load_config
from mozilla_TTS_utils.vocoder.utils.generic_utils import setup_generator
from encoder.mozilla_tts.model import SpeakerEncoder

from validation.paths import *

USE_CUDA = True


def tts(model, vocoder_model, text, CONFIG, use_cuda, ap, use_gl, speaker_fileid, speaker_embedding=None):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, _, _ = synthesis(model, text, CONFIG, use_cuda, ap, speaker_fileid, None, False, CONFIG.enable_eos_bos_chars, use_gl, speaker_embedding=speaker_embedding)
    if CONFIG.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T.unsqueeze(0)).T
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
    if use_cuda and not use_gl:
        waveform = waveform.cpu()
    if not use_gl:
        waveform = waveform.numpy()
    waveform = waveform.squeeze()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    return waveform


class Inferencer():
    def __init__(self, model_path, vocoder_path, encoder_path,
                 model_config, vocoder_config, encoder_config,
                 speaker_json, use_cuda=True):
        self.speakers_mapping = json.load(open(speaker_json, 'r'))
        num_speakers = len(self.speakers_mapping)

        self.model_config = load_config(model_config)
        self.model_config.forward_attn_mask = True
        symbols = ()
        phonemes = ()
        if 'characters' in self.model_config.keys():
            symbols, phonemes = make_symbols(**self.model_config.characters)
        self.ap = AudioProcessor(**self.model_config.audio)
        num_chars = len(phonemes) if self.model_config.use_phonemes else len(symbols) - 1
        self.model = setup_model(num_chars, num_speakers, self.model_config, 256)
        cp = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(cp['model'])
        self.model.eval()
        self.model.decoder.set_r(cp['r'])
        if use_cuda:
            self.model.cuda()

        self.vocoder_config = load_config(vocoder_config)
        if vocoder_path != "":
            self.vocoder = setup_generator(self.vocoder_config)
            self.vocoder.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"])
            self.vocoder.remove_weight_norm()
            self.vocoder.eval()
            self.vocoder.cuda()
        else:
            self.vocoder = None
        self.use_griffin_lim = vocoder_path == ""

        self.encoder_config = load_config(encoder_config)
        self.encoder = SpeakerEncoder(**self.encoder_config.model)
        self.encoder.load_state_dict(torch.load(encoder_path)['model'])
        self.encoder.eval()
        if use_cuda:
            self.encoder.cuda()


    def get_json_output(self, speaker, text, output_path, save_wavs=True):
        speaker_embedding = None
        SPEAKER_FILEID = ''

        for filename, d in self.speakers_mapping.items():
            if d['name'] == speaker:
                SPEAKER_FILEID = filename
                break
        if SPEAKER_FILEID is not None:
            speaker_embedding = self.speakers_mapping[SPEAKER_FILEID]['embedding']
        else:
            choise_speaker = list(self.speakers_mapping.keys())[0]
            print(" Speaker: ", choise_speaker.split('_')[0], 'was chosen automatically',
                  "(this speaker seen in training)")
            speaker_embedding = self.speakers_mapping[choise_speaker]['embedding']

        if SPEAKER_FILEID.isdigit():
            SPEAKER_FILEID = int(SPEAKER_FILEID)
        else:
            SPEAKER_FILEID = None

        wav = tts(self.model, self.vocoder, text, self.model_config, USE_CUDA, self.ap, self.use_griffin_lim, SPEAKER_FILEID,
                  speaker_embedding=speaker_embedding)


        file_name = text.replace(" ", "_")
        file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
        out_path = os.path.join(output_path, file_name)
        if save_wavs:
            self.ap.save_wav(wav, out_path)
        return wav

    def single_output(self, text, input_path, output_path, save_wavs=True, use_cuda=True):
        mel_spec = self.ap.melspectrogram(self.ap.load_wav(input_path, sr=self.ap.sample_rate)).T
        mel_spec = torch.FloatTensor(mel_spec[None, :,:])
        mel_spec = mel_spec.cuda()
        embedd = self.encoder.compute_embedding(mel_spec)
        embedd = embedd.detach().cpu().numpy()
        wav = tts(self.model, self.vocoder, text, self.model_config, USE_CUDA, self.ap, self.use_griffin_lim,
                  None,
                  speaker_embedding=embedd)
        file_name = text.replace(" ", "_")
        file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
        out_path = os.path.join(output_path, file_name)
        if save_wavs:
            self.ap.save_wav(wav, out_path)
        return wav


if __name__ == '__main__':
    test = Inferencer(GRAPHEME_MODEL_PATH, '', ENCODER_PATH, GRAPHEME_CONFIG_PATH, VOCODER_CONFIG_PATH, ENCODER_CONFIG, SPEAKER_JSON, True)
    test.single_output('Это проверочное предложение синтезировано моделью синтеза речи, сделанной в рамках подготовки диплома',
                       '/home/iref/PycharmProjects/tts-vc/data/val_data/user15/common_voice_ru_19028778.wav',
                       OUT_PATH)



