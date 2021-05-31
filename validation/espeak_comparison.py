from paths import TEXT_PATH
bounds = [0, 20, 80, 140]
import subprocess
import speech_recognition as sr
from correctness_validation import get_transcription
from pathlib import Path
import string
import os
import Levenshtein as L
import pandas as pd
import numpy as np



def get_voice():
    i = 0
    with open(TEXT_PATH, 'r') as tf:
        for line in tf:
            line = line.replace(" ", "_")
            command = f'espeak "{line}" -w ./espeak/{i}.wav -vru -s100'
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()
            i+=1


def get_transcr():
    r = sr.Recognizer()
    lines = []
    comparison = []
    result = []
    with open(TEXT_PATH, 'r') as tf:
        for line in tf:
            lines.append(line.strip())

    for i in range(len(lines)):
        orig = lines[i]
        with sr.AudioFile(f'./espeak/{i}.wav') as source:
            audio = r.record(source)
            try:
                transcr = r.recognize_google(audio, language='ru-RU')
            except sr.UnknownValueError:
                transcr = 'TE'
            except sr.RequestError:
                transcr = 'RE'
            comparison.append({'original_text': orig.strip(), 'transcription': transcr})

    comparison = pd.DataFrame(comparison)
    stats = pd.DataFrame(columns=['Len', 'L dist', 'Share'])

    for ind, row in comparison.iterrows():
        original = row['original_text']
        tr = row['transcription']
        if tr != 'TE':
            leven = L.distance(original, tr)
            share = np.round(leven / len(original), 2)
            stats.loc[ind] = [len(original), leven, share]

    short = ''
    medium = ''
    long = ''
    for i in range(len(bounds) - 1):
        temp = stats.loc[stats['Len'].isin(range(bounds[i], bounds[i + 1]))]
        d_mean = np.round(temp["L dist"].mean(), 2)
        s_mean = np.round(temp["Share"].mean(), 2)
        if i == 0:
            short = f'{d_mean}/{s_mean}'
        if i == 1:
            medium = f'{d_mean}/{s_mean}'
        if i == 2:
            long = f'{d_mean}/{s_mean}'
    result.append({'Short': short, 'Medium': medium, 'Long': long})

    result = pd.DataFrame(result)
    result.to_csv('/run/media/iref/Seagate Expansion Drive/spbu_diploma/c_espeak.csv', index=False, sep=',', decimal='.')
    return result
print(get_transcr())



