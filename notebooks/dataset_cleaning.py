import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
from pydub import AudioSegment
import glob

wav_data_path = Path('../data/val_data/')
embeddings_path = Path('../data/val_speaker_embeddings')
mono_path = Path('/home/iref/Desktop/Diploma/mono/')

def make_speakers_json():
    speaker_mapping = {}
    for folder in os.listdir(wav_data_path):
        for file in os.listdir(wav_data_path / folder):
            speaker_mapping[file] = {}
            speaker_mapping[file]['name'] = folder
            try:
                emb = np.load(embeddings_path / folder / file.replace('.wav', '.npy'))
            except:
                continue
            speaker_mapping[file]['embedding'] = emb.flatten().tolist()

        print(len(speaker_mapping.keys()))

        with open('../data/val_speaker_embeddings/speaker.json', 'w') as fp:
            json.dump(speaker_mapping, fp)

def russian_tts(root_path):
    items = []
    for folder in os.listdir(root_path):
        with open(f'{root_path}/{folder}/metadata.csv', 'r') as ttf:
            for line in ttf:
                cols = line.split('|')
                wav_file = os.path.join(root_path, folder, cols[0])
                if not wav_file.endswith('.wav'):
                    wav_file+='.wav'
                    #print(wav_file)
                text = cols[1]
                eng_symbols = re.search('[a-zA-Z]', text)
                if eng_symbols:
                    print(text)
                    print(eng_symbols.group(0), eng_symbols.start(), wav_file)
                    print('#'*32)
                items.append([wav_file, text,  folder])
    return items


def mono_conversion(output_path):
    for folder in os.listdir(wav_data_path):
        os.mkdir(output_path / folder)
        wavs = [f for f in os.listdir(wav_data_path / folder) if f.endswith('.wav')]
        for f in wavs:
            sound = AudioSegment.from_wav(wav_data_path / folder / f)
            sound = sound.set_channels(1)
            sound.export(output_path / folder / f, format="wav")

#import shutil
#for folder in os.listdir(wav_data_path):
    #shutil.copyfile(wav_data_path / folder / 'metadata.csv', mono_path / folder / 'metadata.csv')
#mono_conversion(mono_path)
# print(len(russian_tts(wav_data_path)))
make_speakers_json()

