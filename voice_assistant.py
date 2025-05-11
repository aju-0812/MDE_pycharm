import os
import queue
import sounddevice as sd
import json
from vosk import Model, KaldiRecognizer
from gtts import gTTS
from playsound import playsound
import tempfile


class VoiceAssistant:
    def __init__(self, language="en"):
        self.language = language.lower()
        model_path = {
            "en": "D:/MDE_pycharm/models/vosk-model-small-en-us-0.15",
            "ta": "D:/MDE_pycharm/models/vosk-model-small-en-in-0.4"
        }.get(self.language, "models/vosk-model-small-en-us-0.15")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Vosk model not found at {model_path}")

        self.model = Model(model_path)
        self.q = queue.Queue()
        self.device = None

    def recognize_from_microphone(self):
        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.q.put(bytes(indata))

        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=callback):
            print(f"🎙️ Speak something in {self.language.upper()}...")
            rec = KaldiRecognizer(self.model, 16000)
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    return result.get("text", "")

    def speak(self, object_name, depth):
        if self.language == "en":
            text = f"Detected {object_name}, approximately {depth:.2f} meters away"
        elif self.language == "ta":
            text = f"{object_name} கண்டறியப்பட்டது, சுமார் {depth:.2f} மீட்டர் தொலைவில் உள்ளது"
        else:
            text = f"{object_name} detected at {depth:.2f} meters"

        print(f"[Voice Assistant]: {text}")
        tts = gTTS(text=text, lang=self.language)
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            tts.save(fp.name)
            playsound(fp.name)
