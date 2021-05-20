import argparse
import json
import os
import string
import time
import numpy as np
import torch

from mozilla_TTS_utils.tts_generic_utils import setup_model
from mozilla_TTS_utils.synthesis import synthesis
from mozilla_TTS_utils.text.symbols import make_symbols, phonemes, symbols
from mozilla_TTS_utils.audio import AudioProcessor
from mozilla_TTS_utils.io import load_config
from mozilla_TTS_utils.vocoder.utils.generic_utils import setup_generator, interpolate_vocoder_input

OUT_PATH = '../tests-audios_3/'
# os.makedirs(OUT_PATH, exist_ok=True)
MODEL_PATH = '/home/iref/Desktop/best_model.pth.tar'
CONFIG_PATH = '/home/iref/Desktop/config.json'
SPEAKER_JSON = '/home/iref/PycharmProjects/tts-vc/data/preprocessed_mozilla/speaker.json'
VOCODER_PATH = '/home/iref/Desktop/vocoder/best_model_200.pth.tar'
VOCODER_CONFIG_PATH = '/home/iref/Desktop/vocoder/config.json'
USE_CUDA = True


def tts(model, vocoder_model, text, CONFIG, use_cuda, ap, use_gl, speaker_fileid, speaker_embedding=None):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, _, _ = synthesis(model, text, CONFIG, use_cuda, ap, speaker_fileid, None, False, CONFIG.enable_eos_bos_chars, use_gl, speaker_embedding=speaker_embedding)
    if CONFIG.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T.unsqueeze(0)).T
    if not use_gl:
        #mel_postnet_spec = interpolate_vocoder_input(1.5, mel_postnet_spec)
        waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
    if use_cuda and not use_gl:
        waveform = waveform.cpu()
    if not use_gl:
        waveform = waveform.numpy()
    waveform = waveform.squeeze()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    #print(" > Run-time: {}".format(time.time() - t_1))
    #print(" > Real-time factor: {}".format(rtf))
    #print(" > Time per step: {}".format(tps))
    return waveform



def get_wav_output(speaker, speaker_json, text,
                   model_path, model_config_path,
                   vocoder_path, vocoder_config_path,
                   output_path, save_wavs = True):
    C = load_config(model_config_path)
    C.forward_attn_mask = True
    ap = AudioProcessor(**C.audio)
    symbols = ()
    phonemes = ()
    if 'characters' in C.keys():
        symbols, phonemes = make_symbols(**C.characters)

    speaker_embedding = None
    speaker_embedding_dim = None
    num_speakers = 0
    SPEAKER_FILEID = ''

    speaker_mapping = json.load(open(speaker_json, 'r'))
    num_speakers = len(speaker_mapping)
    for filename, d in speaker_mapping.items():
        if d['name'] == speaker:
            SPEAKER_FILEID = filename
            break

    if SPEAKER_FILEID is not None:
        speaker_embedding = speaker_mapping[SPEAKER_FILEID]['embedding']
    else:  # if speaker_fileid is not specificated use the first sample in speakers.json
        choise_speaker = list(speaker_mapping.keys())[0]
        print(" Speaker: ", choise_speaker.split('_')[0], 'was chosen automatically',
              "(this speaker seen in training)")
        speaker_embedding = speaker_mapping[choise_speaker]['embedding']
    speaker_embedding_dim = len(speaker_embedding)

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, num_speakers, C, speaker_embedding_dim)
    cp = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model'])
    model.eval()

    if USE_CUDA:
        model.cuda()

    model.decoder.set_r(cp['r'])

    # load vocoder model
    if vocoder_path != "":
        VC = load_config(vocoder_config_path)
        vocoder_model = setup_generator(VC)
        vocoder_model.load_state_dict(torch.load(vocoder_path, map_location="cpu")["model"])
        vocoder_model.remove_weight_norm()
        if USE_CUDA:
            vocoder_model.cuda()
        vocoder_model.eval()
    else:
        vocoder_model = None
        VC = None
    use_griffin_lim = vocoder_path == ""
    if not C.use_external_speaker_embedding_file:
        if SPEAKER_FILEID.isdigit():
            SPEAKER_FILEID = int(SPEAKER_FILEID)
        else:
            SPEAKER_FILEID = None
    else:
        SPEAKER_FILEID = None


    wav = tts(model, vocoder_model, text, C, USE_CUDA, ap, use_griffin_lim, SPEAKER_FILEID, speaker_embedding=speaker_embedding)
    file_name = text.replace(" ", "_")
    file_name = file_name.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
    #file_name = f'{d["name"]}'
    out_path = os.path.join(output_path, file_name)
    if save_wavs:
        ap.save_wav(wav, out_path)
    return wav



