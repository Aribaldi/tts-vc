import argparse
from validation.inference_test import Inferencer
from validation.paths import *

def main(args):
    if args.model_type == 'g':
        if args.use_vocoder:
            inf = Inferencer(GRAPHEME_MODEL_PATH, VOCODER_PATH, ENCODER_PATH, GRAPHEME_CONFIG_PATH,
                             VOCODER_CONFIG_PATH, ENCODER_CONFIG, SEEN_SPEAKERS_JSON, True)
        else:
            inf = Inferencer(GRAPHEME_MODEL_PATH, '', ENCODER_PATH, GRAPHEME_CONFIG_PATH, VOCODER_CONFIG_PATH,
                             ENCODER_CONFIG, SEEN_SPEAKERS_JSON, True)
    else:
        if args.use_vocoder:
            inf = Inferencer(PHONEME_MODEL_PATH, VOCODER_PATH, ENCODER_PATH, PHONEME_CONFIG_PATH,
                             VOCODER_CONFIG_PATH, ENCODER_CONFIG, SEEN_SPEAKERS_JSON, True)
        else:
            inf = Inferencer(PHONEME_MODEL_PATH, '', ENCODER_PATH, PHONEME_CONFIG_PATH,
                             VOCODER_CONFIG_PATH, ENCODER_CONFIG, SEEN_SPEAKERS_JSON, True)
    print('Generating audio...')
    inf.single_output(args.text, args.inp_wav, args.output_path)
    print('Done')




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('inp_wav', help='Path to input wav file')
    parser.add_argument('text', help='Input text')
    parser.add_argument('--use_vocoder', help='Turn on MelBand vocoder; GL algorithm is used by default',
                        action='store_true')
    parser.add_argument('model_type', help='Choose whether grapheme or morpheme model to use: g - grapheme, p - phoneme')
    parser.add_argument('-o', '--output_path', help='Output wav path')
    args = parser.parse_args()

    main(args)

