import os
from pathlib import Path
import pandas as pd
import glob
import shutil
from pydub import AudioSegment

ROOT_PATH = Path('/home/iref/tts-vc/data/speaker_encoder_data/')
VAL_PATH = Path('/home/iref/Datasets/cv-corpus-5.1-2020-06-22/ru/')
VAL_WAV = VAL_PATH / 'clips'
OUT_DIR = Path('/home/iref/PycharmProjects/tts-vc/data/val_data')

all_train_wavs = glob.glob(str(ROOT_PATH) + '/**/*.wav')
all_train_wavs = [x for x in all_train_wavs if 'common_voice_ru' in x]
all_train_wavs = list(map(Path, all_train_wavs))
all_train_wavs = list(map(lambda x: x.stem, all_train_wavs))
all_train_wavs = list(map(lambda x: x + '.mp3', all_train_wavs))
print('Num of used wavs:', len(all_train_wavs))
val_df = pd.read_csv(VAL_PATH / 'train.tsv', sep='\t')
print('Num of new wavs:', len(val_df))
val_wavs = val_df['path'].to_list()
print('Speakers num in new dataset:', len(val_df['client_id'].unique()))

intersection = set(val_wavs).intersection(all_train_wavs)
print(len(intersection))
diff = set(val_wavs).difference(all_train_wavs)
print(len(diff))

speakers_list = []
for file in intersection:
    row = val_df.loc[val_df['path'] == file]
    speakers_list.append(row['client_id'].item())

speakers_list = set(speakers_list)
print(len(speakers_list))

clear = val_df[~val_df['client_id'].isin(speakers_list)]
print(len(clear['sentence'].unique()) == len(clear))

# i = 1
# for cl in clear['client_id'].unique():
#     dest = OUT_DIR / f'user{i}'
#     os.makedirs(dest, exist_ok=True)
#     temp = clear.loc[clear['client_id'] == cl]
#     for filename in temp['path'].unique():
#         #shutil.copyfile(VAL_WAV / filename, dest / filename)
#         t = AudioSegment.from_mp3(VAL_WAV / filename)
#         t = t.set_channels(1)
#         t.set_frame_rate(16000)
#         t.export(dest / str(filename[:-3] + 'wav'), format='wav')
#
#     meta = temp.drop(temp.columns.difference(['path','sentence']), 1)
#     meta['path'] = meta['path'].apply(lambda x: x.replace('mp3', 'wav'))
#     meta.to_csv(dest / 'metadata.csv', index=False, sep='\t', header=None)
#     i += 1

