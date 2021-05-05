from encoder.real_time_vc.preprocess import _init_preprocess_dataset, _preprocess_speaker_dirs
from pathlib import Path
from encoder.real_time_vc.train import train

dataset_name = 'speaker_encoder_data'
datasets_root = Path('../data')
out_dir = datasets_root / 'preprocessed'
#full_path = datasets_root / dataset_name
skip_existing = True
models_dir = Path('../models_and_weights')


def data_prep():
    dataset_root, logger = _init_preprocess_dataset(dataset_name, datasets_root, out_dir)
    speaker_dirs = list(dataset_root.glob("*"))
    _preprocess_speaker_dirs(speaker_dirs, dataset_name, datasets_root, out_dir, "wav",
                             skip_existing, logger)

def train_wrapper():
    train('first', out_dir, models_dir, umap_every=0, save_every=1, backup_every=0,
          vis_every=1, force_restart=True, visdom_server='http://localhost', no_visdom=False)


if __name__ == '__main__':
    train_wrapper()