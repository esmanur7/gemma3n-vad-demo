import torch
import sounddevice as sd
import numpy as np
import queue
import threading
import time

from transformers import AutoProcessor, AutoModelForImageTextToText, TextIteratorStreamer

device = "cuda"

vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
(get_speech_timestamps, save_vad_model, read_audio, *rest) = utils

MODEL_ID = "google/gemma-3n-E2B-it"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True
).eval()

history = []
audio_q = queue.Queue()

SAMPLING_RATE = 16000
BLOCK_SIZE = 1024          # ~64ms
CHANNELS = 1

def audio_callback(indata, frames, time_info, status):
    audio_q.put(indata.copy())
    

def vad_worker():
    global history

    print("\nKonuşabilirsin..\n")

    buffer = []

    is_speaking = False
    last_voice_time = time.time()

    while True:
        audio_block = audio_q.get()
        buffer.append(audio_block)

        audio_np = np.concatenate(buffer, axis=0)
        audio_torch = torch.from_numpy(audio_np.T)

        # konuşma var mı?
        speech = get_speech_timestamps(audio_torch, vad_model, sampling_rate=SAMPLING_RATE)

        if speech:
            if not is_speaking:
                print("Konuşma algılandı!")
            is_speaking = True
            last_voice_time = time.time()

        # konuşma bittiyse (2s sessizlik)
        if is_speaking and (time.time() - last_voice_time > 2.0):
            print("Konuşma bitti modele gönderiliyor...\n")
            full_audio = np.concatenate(buffer, axis=0)
            buffer = []
            is_speaking = False

            run_gemma(full_audio.flatten())


def run_gemma(audio_np):
    global history

    print("Model..\n")

    messages = history + [
        {
            "role": "user",
            "content": [
                {"type":"audio", "audio": audio_np},
                {"type":"text", "text":"The speech you hear is the user's message. Respond as if you are talking to them directly. Respond naturally. Respond in Turkish."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)

    start_time = time.time()
    first_token_time = None

    def run():
        model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            streamer=streamer
        )

    thread = threading.Thread(target=run)
    thread.start()

    output = ""

    for token in streamer:
        if first_token_time is None:
            first_token_time = time.time()
            ttft = first_token_time - start_time
            print(f"TTFT = {ttft:.2f} saniye\n")

        print(token, end="", flush=True)
        output += token

    print("\n")

    history.append({"role":"user","content":[{"type":"audio","audio":audio_np}]})
    history.append({"role":"assistant","content":[{"type":"text","text":output}]})


stream = sd.InputStream(
    channels=CHANNELS,
    samplerate=SAMPLING_RATE,
    blocksize=BLOCK_SIZE,
    callback=audio_callback
)
stream.start()

threading.Thread(target=vad_worker, daemon=True).start()
