#!/bin/sh
apt-get install git
git clone https://github.com/Aribaldi/tts-vc


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MeVw8FlxWSKUpcQbqTrr-PPilANHJDZb' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MeVw8FlxWSKUpcQbqTrr-PPilANHJDZb" -O preprocessed_mozilla.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1AtKExpkx3_ts6no9vaf1TKv1Uv7eX17f' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1AtKExpkx3_ts6no9vaf1TKv1Uv7eX17f" -O speaker_encoder_data.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lFYuEmGerPjQPYrxOr84Ooa2q1lhnuwG' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lFYuEmGerPjQPYrxOr84Ooa2q1lhnuwG" -O phoneme_cache.zip && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TkuJkxs5nZwyXx6GqwZp4wL20Wm2_Fem' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TkuJkxs5nZwyXx6GqwZp4wL20Wm2_Fem" -O speaker.json && rm -rf /tmp/cookies.txt

apt-get install unzip
unzip preprocessed_mozilla.zip -d tts-vc/data/
unzip speaker_encoder_data.zip -d tts-vc/data/
unzip phoneme_cache.zip -d tts-vc/data/tts_output

pip install --user -r requirements.txt
apt-get install espeak

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1bsziFa5rjjSR_mnU0JvZajRiDDuceEzr' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1bsziFa5rjjSR_mnU0JvZajRiDDuceEzr" -O test_sentences.txt && rm -rf /tmp/cookies.txt


wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aKQWynvRnv6MNhFtcOjODb1HvFM02Ko8' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aKQWynvRnv6MNhFtcOjODb1HvFM02Ko8" -O checkpoint_275000.pth.tar && rm -rf /tmp/cookies.txt

https://drive.google.com/file/d/1aKQWynvRnv6MNhFtcOjODb1HvFM02Ko8/view?usp=sharing