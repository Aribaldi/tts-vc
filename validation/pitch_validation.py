import os
import resemblyzer as res
import json
from pathlib import Path
import numpy as np
ORIGS = Path('/home/iref/tts-vc/data/speaker_encoder_data/user10/')
GENERATED = Path('/tests-audios_3/')
encoder = res.VoiceEncoder()


def get_similarity_score(originals_path, generated_path):
    with open(generated_path / 'sentences.json', 'r') as f:
        sents = json.load(f)
    gen_list = []
    orig_list = []

    g_u = np.empty((10, 256))
    o_u = np.empty((10, 256))

    i = 0
    for el in sents:
        path = ''
        if el["name"].endswith('.wav'):
            path = f'{el["name"]}'
        else:
            path = f'{el["name"]}.wav'
        gen_wav = res.preprocess_wav(generated_path / f'{el["name"]}_generated.wav')
        orig_wav = res.preprocess_wav(originals_path / path)

        g_u[i] = encoder.embed_utterance(gen_wav)
        o_u[i] = encoder.embed_utterance(orig_wav)

        gen_list.append(gen_wav)
        orig_list.append(orig_wav)

    original_speaker_embedding = encoder.embed_speaker(orig_list)
    generated_speaker_embedding = encoder.embed_speaker(gen_list)


    result = {}
    result['similarity'] = np.inner(original_speaker_embedding, generated_speaker_embedding)
    result['L2-distance'] = np.linalg.norm(original_speaker_embedding - generated_speaker_embedding)
    return result

def get_same_phrases():
    root_path = '/home/iref/tts-vc/data/speaker_encoder_data/'
    users_list = [f'user{i}' for i in range(1, 31)]
    items = {}
    for user in users_list:
        items[user] = {'text': [], 'files':[]}
        with open(f'{root_path}/{user}/metadata.csv', 'r') as ttf:
            for line in ttf:
                cols = line.split('|')
                wav_file = os.path.join(root_path, user, cols[0])
                if not wav_file.endswith('.wav'):
                    wav_file += '.wav'
                    #print(wav_file)
                text = cols[1].strip()
                text = text.strip(',')
                items[user]['text'].append(text)
                items[user]['files'].append(wav_file)
    for user in users_list:
        diff = set(users_list).difference(user)
        for user2 in diff:
            intersection = set(items[user]['text']).intersection(items[user2]['text'])
            if len(intersection)!=0:
                print(user, user2)
                print(intersection)


def diff_speakers_same_phrases(user1, user2):
    root_path = '/home/iref/tts-vc/data/speaker_encoder_data/'
    phrases1 = {}
    phrases2 = {}
    with open(f'{root_path}/{user1}/metadata.csv', 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, user1, cols[0])
            if not wav_file.endswith('.wav'):
                wav_file += '.wav'
            text = cols[1].strip()
            text = text.strip(',')
            phrases1[text] = wav_file

    with open(f'{root_path}/{user2}/metadata.csv', 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, user2, cols[0])
            if not wav_file.endswith('.wav'):
                wav_file += '.wav'
            text = cols[1].strip()
            text = text.strip(',')
            phrases2[text] = wav_file

    inter = set(phrases1.keys()).intersection(phrases2.keys())

    wavs_1 = []
    wavs_2 = []

    for phr in inter:
        wavs_1.append(res.preprocess_wav(phrases1[phr]))
        wavs_2.append(res.preprocess_wav(phrases2[phr]))

    embedding_1 = encoder.embed_speaker(wavs_1)
    embedding_2 = encoder.embed_speaker(wavs_2)
    print(np.inner(embedding_1, embedding_2), np.linalg.norm(embedding_1 - embedding_2))


print(get_similarity_score(ORIGS, GENERATED, 10))
#print(get_same_phrases()
print(diff_speakers_same_phrases('user15', 'user2'))
