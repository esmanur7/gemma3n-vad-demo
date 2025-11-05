import torch
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
from llama_cpp import Llama
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import numpy as np

sd.default.device = 12   # Mikrofon, hoparlör

#Ses Kaydı
def record(filename="input.wav", duration=5, fs=16000):
    print(" Konuş (kayıt başlıyor)...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    write(filename, fs, audio)
    print("Kayıt tamam!\n")

#VAD (Sessizlik Ayırma)
vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
(get_speech_timestamps, _, _, _, _) = utils

#Whisper (CPU)
asr = WhisperModel("tiny", device="cpu", compute_type="int8")

def speech_to_text(file_path):
    segments, _ = asr.transcribe(
        file_path,
        language="tr",
        beam_size=5
    )
    return "".join([s.text for s in segments]).strip()

#Gemma (GGUF, CPU)
llm = Llama(
    model_path="/home/esma/models/gemma/gemma-2b.gguf",
    n_threads=6,
    n_ctx=2048
)

def chat(history):
    conversation = ""
    for turn in history:
        role = "user" if turn["role"] == "user" else "assistant"
        conversation += f"<start_of_turn>{role}\n{turn['content']}<end_of_turn>\n"
    conversation += "<start_of_turn>assistant\n"

    output = llm(conversation, max_tokens=150, temperature=0.7, stop=["<end_of_turn>"])
    return output["choices"][0]["text"].strip()

#TTS
def speak(text):
    tts = gTTS(text=text, lang="tr")
    tts.save("reply.mp3")
    sound = AudioSegment.from_mp3("reply.mp3")
    play(sound)

#Loop
history = []

while True:
    record(duration=5)

    audio, sr = sf.read("input.wav", dtype="float32", always_2d=True)
    mono = audio.mean(axis=1, keepdims=True)
    wav = torch.from_numpy(mono.T)

    if not get_speech_timestamps(wav, vad_model, sampling_rate=sr):
        print("Konuşma algılanmadı. Tekrar dene.\n")
        continue

    user_text = speech_to_text("input.wav")
    print(f"Sen: {user_text}")

    history.append({"role": "user", "content": user_text})

    print(" Cevap hazırlanıyor\n")
    reply = chat(history)
    print(f"Asistan:\n{reply}\n")

    history.append({"role": "assistant", "content": reply})
    speak(reply)
