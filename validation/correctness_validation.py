import speech_recognition as sr
from pathlib import Path
import pandas as pd
import string
import numpy as np
import Levenshtein as L

from paths import *
from inference_test import Inferencer


bounds = [0, 20, 80, 140]


def pipeline(speakers_list, save_audio_path):
    if len(speakers_list) == 30:
        inf = Inferencer(GRAPHEME_MODEL_PATH, VOCODER_PATH, ENCODER_PATH, GRAPHEME_CONFIG_PATH, VOCODER_CONFIG_PATH, ENCODER_CONFIG,
                         SEEN_SPEAKERS_JSON)
    else:
        inf = Inferencer(GRAPHEME_MODEL_PATH, VOCODER_PATH, ENCODER_PATH, GRAPHEME_CONFIG_PATH, VOCODER_CONFIG_PATH, ENCODER_CONFIG,
                         UNSEEN_SPEAKERS_JSON)
    result = []
    for speaker in speakers_list:
        user_stats_df = pd.DataFrame(columns=['User', 'Phrase len', 'L Dist', 'Share'])
        with open(TEXT_PATH, 'r') as tf:
            for line in tf:
                line = line.strip()
                inf.get_json_output(speaker, line, save_audio_path)

        comparison = pd.DataFrame(get_transcription(AUDIO_PATH))
        for ind, row in comparison.iterrows():
            original = row['original_text']
            tr = row['transcription']
            if tr != 'TE':
                leven = L.distance(original, tr)
                share = np.round(leven / len(original), 2)
                user_stats_df.loc[ind] = [speaker, len(original), leven, share]

        short = ''
        medium = ''
        long = ''
        for i in range(len(bounds) - 1):
            temp = user_stats_df.loc[user_stats_df['Phrase len'].isin(range(bounds[i], bounds[i + 1]))]
            d_mean = np.round(temp["L Dist"].mean(), 2)
            s_mean = np.round(temp["Share"].mean(), 2)
            if i == 0:
                short = f'{d_mean}/{s_mean}'
            if i == 1:
                medium = f'{d_mean}/{s_mean}'
            if i == 2:
                long = f'{d_mean}/{s_mean}'
        result.append({'User': speaker, 'Short': short, 'Medium': medium, 'Long':long})


    res = pd.DataFrame(result)
    res.to_csv('/run/media/iref/Seagate Expansion Drive/spbu_diploma/test.csv')
    print(res)


def get_transcription(input_audio_path):
    result = []
    with open(input_audio_path / 'text2.txt', 'r') as f:
        for line in f:
            temp = line.replace(" ", "_")
            temp = temp.strip()
            temp = temp.translate(str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
            r = sr.Recognizer()
            with sr.AudioFile(str(input_audio_path / temp)) as source:
                audio = r.record(source)
                try:
                    transcr = r.recognize_google(audio, language='ru-RU')
                except sr.UnknownValueError:
                    transcr = 'TE'
                except sr.RequestError:
                    transcr = 'RE'
                result.append({'original_text':line.strip(), 'transcription': transcr})
        return result


def get_overall_stats(df_path):
    df = pd.read_csv(df_path, index_col=0)
    for l in ['Short', 'Medium', 'Long']:
        temp = df[l].to_list()
        L_d = []
        for el in temp:
            buf = el.split('/')
            L_d.append(float(buf[0]))
        print(np.mean(L_d))

if __name__ == '__main__':
    print(pipeline([f'user{i}' for i in range(1, 6)], AUDIO_PATH))
    get_overall_stats('/run/media/iref/Seagate Expansion Drive/spbu_diploma/test.csv')




