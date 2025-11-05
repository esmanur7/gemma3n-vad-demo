!pip install gTTS torchaudio unsloth transformers --quiet
import torch, torchaudio
from gtts import gTTS
from IPython.display import Audio, display
from unsloth import FastModel
from transformers import TextStreamer

#TTS
text_in = "Merhaba, bu bir test sesidir. Sistem şu anda çalışıyor."
tts = gTTS(text=text_in, lang='tr')
tts.save("audio.mp3")
display(Audio("audio.mp3"))

#16kHz(VAD uyumlu )
wav, sr = torchaudio.load("audio.mp3")
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000
torchaudio.save("audio_16k.wav", wav, sr)

#silero vad
vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

wav, sr = torchaudio.load("audio_16k.wav")
speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr)
vad_text = (f"Bu ses dosyasında {len(speech_timestamps)} konuşma bölümü tespit edildi."
            if speech_timestamps else "Bu ses dosyasında konuşma tespit edilmedi.")
print(f"vad: {vad_text}\n")

#gemma3n
from unsloth import FastModel
from transformers import TextStreamer
import torch


model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    dtype=torch.float32,
    load_in_4bit=True,
    device_map={"": "cpu"},   # model tamamen CPU'da
)


messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": "audio_16k.wav"},
        {"type": "text", "text": f"{vad_text} Bu ses ne hakkında? Lütfen Türkçe olarak açıkla."}
    ]
}]

# tokenizer
tokenized_inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt"
)

if isinstance(tokenized_inputs, torch.Tensor):
    inputs = {"input_ids": tokenized_inputs}
else:
    inputs = tokenized_inputs

for k, v in inputs.items():
    inputs[k] = v.to(model.device)

streamer = TextStreamer(tokenizer, skip_prompt=True)

print("model cevabı: \n")

_ = model.generate(
    **inputs,
    max_new_tokens=128,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
    streamer=streamer
)

'''
             OUTPUT:
Bu ses dosyasında 2 konuşma bölümü tespit edildiğini ve bununla ilgili olası açıklamaları ve anlamlarını açıklayalım: 
**Olayın Şekli:** 
* **Yanlış Tanıma:** Ses dosyasındaki iki bölümün, gerçekte bir konuşmanın *partikülatör* (sözlük) veya *temsil* olarak algılanması. Yani, o ses, aslında bir konuşma değil, 
bir sonraki konuşmanın başlangıcını temsil ediyor. 
***Çözüm:** Ses sistemi, dil modelinin yetersizliği nedeniyle bu tür bir hata yapmıştır
'''
