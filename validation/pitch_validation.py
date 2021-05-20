import os
import resemblyzer as res
import json
from pathlib import Path
import numpy as np
ROOT_PATH = Path('/home/iref/tts-vc/data/speaker_encoder_data/')
encoder = res.VoiceEncoder()
import pandas as pd
from inference_test import OUT_PATH, MODEL_PATH, CONFIG_PATH, SPEAKER_JSON, VOCODER_PATH, VOCODER_CONFIG_PATH, USE_CUDA
from inference_test import get_wav_output


def pitch_val_pipeline():
    result = pd.DataFrame(columns=['First', 'Second', 'Similarity bw originals', 'L2 bw originals', 'User1_gen_sim',
                                   'User1_gen_L2', 'User2_gen_sim', 'User2_gen_L2'])
    users_df = get_same_phrases()
    result['First'] = users_df['First'].to_list()
    result['Second'] = users_df['Second'].to_list()

    originals_sim = []
    originals_L2 = []
    u1_sim = []
    u1_L2 = []
    u2_sim = []
    u2_L2 = []

    for index, row in list(users_df.iterrows()):
        o_sim, o_L2, u1_gen_sim, u1_gen_L2, u2_gen_sim, u2_gen_L2 = get_metrics(row['First'], row['Second'])
        originals_sim.append(o_sim)
        originals_L2.append(o_L2)
        u1_sim.append(u1_gen_sim)
        u1_L2.append(u1_gen_L2)
        u2_sim.append(u2_gen_sim)
        u2_L2.append(u2_gen_L2)


    result['Similarity bw originals'] = originals_sim
    result['L2 bw originals'] = originals_L2
    result['User1_gen_sim'] = u1_sim
    result['User1_gen_L2'] = u1_L2
    result['User2_gen_sim'] = u2_sim
    result['User2_gen_L2'] = u2_L2
    print(result.mean(axis=0))
    result.to_csv('/home/iref/PycharmProjects/tts-vc/validation/pitch_val_unseen.csv')
    return result




def get_same_phrases(mode='val'):
    root_path = ROOT_PATH
    #users_list = [f'user{i}' for i in range(1, 31)]
    users_list = [x for x in os.listdir(root_path) if len(x) == 3]
    print(users_list)
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

    temp = []
    for usr in users_list:
        temp.extend(items[usr]['text'])
    t = set(temp)
    print(len(temp))
    print(len(t))

    used = []
    result = pd.DataFrame(columns=['First', 'Second', 'Intersection_len'])
    firsts = []
    seconds = []
    ints = []
    for user in users_list:
        used.append(user)
        for user2 in users_list:
            if user2 not in used:
                intersection = set(items[user]['text']).intersection(items[user2]['text'])
                if len(intersection) > 10:
                    firsts.append(user)
                    seconds.append(user2)
                    ints.append(len(intersection))
    result['First'] = firsts
    result['Second'] = seconds
    result['Intersection_len'] = ints

    return result



def get_metrics(user1, user2, mode='val'):
    phrases1 = {}
    phrases2 = {}
    with open(f'{ROOT_PATH}/{user1}/metadata.csv', 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(ROOT_PATH, user1, cols[0])
            if not wav_file.endswith('.wav'):
                wav_file += '.wav'
            text = cols[1].strip()
            text = text.strip(',')
            phrases1[text] = wav_file

    with open(f'{ROOT_PATH}/{user2}/metadata.csv', 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(ROOT_PATH, user2, cols[0])
            if not wav_file.endswith('.wav'):
                wav_file += '.wav'
            text = cols[1].strip()
            text = text.strip(',')
            phrases2[text] = wav_file

    inter = set(phrases1.keys()).intersection(phrases2.keys())
    inter = list(inter)
    inter = np.random.choice(inter, 10)

    wavs_1 = []
    wavs_2 = []
    generated_1 = []
    generated_2 = []

    for phr in inter:
        wavs_1.append(res.preprocess_wav(phrases1[phr]))
        wavs_2.append(res.preprocess_wav(phrases2[phr]))
        generated_1.append(get_wav_output(user1, phr, MODEL_PATH, CONFIG_PATH, VOCODER_PATH, VOCODER_CONFIG_PATH,
                                          OUT_PATH, save_wavs=False))
        generated_2.append(get_wav_output(user2, phr, MODEL_PATH, CONFIG_PATH, VOCODER_PATH, VOCODER_CONFIG_PATH,
                                          OUT_PATH, save_wavs=False))

    generated_1 = list(map(res.preprocess_wav, generated_1))
    generated_2 = list(map(res.preprocess_wav, generated_2))

    o_embed_1 = encoder.embed_speaker(wavs_1)
    o_embed_2 = encoder.embed_speaker(wavs_2)

    g_embed_1 = encoder.embed_speaker(generated_1)
    g_embed_2 = encoder.embed_speaker(generated_2)

    return np.inner(o_embed_1, o_embed_2),  np.linalg.norm(o_embed_1 - o_embed_2), \
            np.inner(g_embed_1, o_embed_1), np.linalg.norm(o_embed_1 - g_embed_1), \
            np.inner(g_embed_2, o_embed_2), np.linalg.norm(o_embed_2 - g_embed_2)


#print(get_same_phrases())
#print(pitch_val_pipeline())

frame = np.round(pd.read_csv('/home/iref/PycharmProjects/tts-vc/validation/pitch_val_unseen.csv', index_col=0), 2)
print(frame.columns)
frame.columns = ['UserA', 'UserB', 'ABsim', 'L2betweenoriginals', 'Ageneratedsim',
                                   'L2 distance between user A and generated speaker', 'Bgeneratedsim',
                                   'L2 distance between user A and generated speaker']
sim_frame = frame[['UserA', 'UserB', 'ABsim', 'Ageneratedsim', 'Bgeneratedsim']]
overall_frame = sim_frame.median()
print(overall_frame)
#sim_frame.to_csv('/run/media/iref/Seagate Expansion Drive/spbu_diploma/sim_frame_unseen.csv', sep=',', decimal='.', index=False)
#frame.to_csv('/home/iref/PycharmProjects/tts-vc/validation/test_frame.csv', sep='\t', decimal=',', index=False)
