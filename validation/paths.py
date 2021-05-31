from pathlib import Path

ROOT_PATH = Path('/home/iref/tts-vc/data/speaker_encoder_data/')
TEXT_PATH = Path('/run/media/iref/Seagate Expansion Drive/correctness_tests/text2.txt')
AUDIO_PATH = Path('/run/media/iref/Seagate Expansion Drive/correctness_tests/')
SEEN_SPEAKERS_JSON = Path('/data/train_speaker_embeddings/speaker.json')
UNSEEN_SPEAKERS_JSON = Path('/home/iref/PycharmProjects/tts-vc/data/val_speaker_embeddings/speaker.json')

VOCODER_PATH = '/home/iref/PycharmProjects/tts-vc/models_and_weights/melgan_best_model.pth.tar'
VOCODER_CONFIG_PATH = '/home/iref/PycharmProjects/tts-vc/models_and_weights/melgan_config.json'
ENCODER_PATH = '/home/iref/PycharmProjects/tts-vc/models_and_weights/encoder_best_model.pth.tar'
ENCODER_CONFIG = '/home/iref/PycharmProjects/tts-vc/models_and_weights/encoder_config.json'
OUT_PATH = '/home/iref/PycharmProjects/tts-vc/tests-audios/'
GRAPHEME_MODEL_PATH = '/home/iref/PycharmProjects/tts-vc/models_and_weights/g_tacotron_best_model.pth.tar'
GRAPHEME_CONFIG_PATH = '/home/iref/PycharmProjects/tts-vc/models_and_weights/g_tacotron_config.json'
SPEAKER_JSON = '/home/iref/PycharmProjects/tts-vc/data/train_speaker_embeddings/speaker.json'
PHONEME_MODEL_PATH = '/home/iref/PycharmProjects/tts-vc/models_and_weights/checkpoint_220000.pth.tar'
PHONEME_CONFIG_PATH = '/home/iref/PycharmProjects/tts-vc/models_and_weights/config.json'
#VOCODER_PATH = ""