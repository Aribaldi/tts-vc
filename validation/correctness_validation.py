import speech_recognition as sr
import os
from pathlib import Path
import json
import pandas as pd

AUDIO_PATH = Path('../tests-audios_3')

def get_transcription(input_audio_path):


    result = []
    with open(input_audio_path / 'sentences.json', 'r') as f:
        sents = json.load(f)
        r = sr.Recognizer()
        for sent in sents:
            with sr.AudioFile(str(input_audio_path / sent['name']) + '_generated.wav') as source:
                audio = r.record(source)
                try:
                    transcr = r.recognize_google(audio, language='ru-RU')
                except sr.UnknownValueError:
                    transcr = 'TE'
                except sr.RequestError:
                    transcr = 'RE'
                result.append({'name': sent['name'],'original_text':sent['text'], 'transcription': transcr})
    return result

print(get_transcription(AUDIO_PATH))
