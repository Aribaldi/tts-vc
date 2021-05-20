import speech_recognition as sr
import os
from pathlib import Path
import json
import pandas as pd
import gensim
import string
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

TEXT_PATH = Path('/run/media/iref/Seagate Expansion Drive/correctness_tests/text.txt')
AUDIO_PATH = Path('/run/media/iref/Seagate Expansion Drive/correctness_tests/')
SEEN_SPEAKERS_JSON = Path('/home/iref/PycharmProjects/tts-vc/data/preprocessed_mozilla/speaker.json')
UNSEEN_SPEAKERS_JSON = Path('/home/iref/PycharmProjects/tts-vc/data/val_speaker_embeddings/speaker.json')
from inference_test import get_wav_output
from inference_test import MODEL_PATH, CONFIG_PATH, VOCODER_PATH, VOCODER_CONFIG_PATH


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print ("module %s loaded" % module_url)

def pipeline(speakers_list, save_audio_path):

    acc = []
    dists = []
    for speaker in speakers_list:
        with open(TEXT_PATH, 'r') as tf:
            for line in tf:
                line = line.strip()
                get_wav_output(speaker, UNSEEN_SPEAKERS_JSON, line, MODEL_PATH, CONFIG_PATH, VOCODER_PATH, VOCODER_CONFIG_PATH,
                               save_audio_path, save_wavs=True)

        comparison = pd.DataFrame(get_transcription(AUDIO_PATH))
        right_share = []
        l2s = []
        for ind, row in comparison.iterrows():
            original = set(row['original_text'])
            orig_embed = model([original])[0]
            tr = set(row['transcription'])
            if tr != 'TE':
                inter = original.intersection(tr)
                right_share.append(len(inter) / len(original))
                tr_embed = model([tr])[0]
                l2s.append(np.linalg.norm(orig_embed, tr_embed))

        acc.append(np.mean(right_share))
        dists.append(np.mean(dists))

    result = pd.DataFrame(columns=['User', 'Accuracy'])
    result['User'] = speakers_list
    result['Accuracy'] = acc
    result['USE distances'] = dists
    result.to_csv('/run/media/iref/Seagate Expansion Drive/spbu_diploma/c_check.csv')
    return result






def get_transcription(input_audio_path):
    result = []
    with open(input_audio_path / 'text.txt', 'r') as f:
        for line in f:
            line = line.strip()
            r = sr.Recognizer()
            temp = line.replace(" ", "_")
            with sr.AudioFile(str(input_audio_path / temp) + '.wav') as source:
                audio = r.record(source)
                try:
                    transcr = r.recognize_google(audio, language='ru-RU')
                except sr.UnknownValueError:
                    transcr = 'TE'
                except sr.RequestError:
                    transcr = 'RE'
                result.append({'original_text':line, 'transcription': transcr})
        return result


print(pipeline([f'user{i}' for i in range(1, 6)], AUDIO_PATH))
#print(get_transcription(AUDIO_PATH))
