"""
!pip install -q unsloth transformers accelerate bitsandbytes torchaudio soundfile sentencepiece

from google.colab import files
uploaded = files.upload()

!apt -y install ffmpeg
!ffmpeg -y -i "WhatsApp Audio 2025-11-06 at 15.02.48.mp4" -ac 1 -ar 16000 user1.wav
!ffmpeg -y -i "WhatsApp Audio 2025-11-06 at 15.02.48(1).mp4" -ac 1 -ar 16000 user2.wav

"""

import unsloth
from unsloth import FastLanguageModel
import torch, torchaudio, soundfile as sf

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# MODEL
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    load_in_4bit=True,
    dtype=torch.float32,
    max_seq_length=4096,
    device_map="auto",
    offload_folder="offload",      # VRAM yetmezse CPU'ya taşır
)
model = FastLanguageModel.for_inference(model)

# VAD
vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
(get_speech_timestamps, _, _, _, collect_chunks) = utils

def has_speech(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return len(get_speech_timestamps(wav, vad_model, sampling_rate=sr)) > 0

history = []

def run_audio(path):
    print(f"\nSes işleniyor : {path}")

    if not has_speech(path):
        print("Konuşma algılanmadı.\n")
        return

    history.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio": path},
            {"type": "text", "text": "Doğal bir şekilde yanıtla."}
        ]
    })
    print("History uzunluğu:", len(history))

    print("Model..\n")

    encoded = tokenizer.apply_chat_template(
        history,
        add_generation_prompt=True,
        tokenize=True,                       
        return_tensors="pt"
    ).to(model.device)                       

    with torch.no_grad():
        output = model.generate(
            input_ids=encoded,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
        )

    #reply = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = output[0][encoded.shape[1]:]   # sadece yeni tokenlar
    reply = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print("Asistan:", reply, "\n")

    history.append({
        "role": "assistant",
        "content": [{"type": "text", "text": reply}]
    })

run_audio("user1.wav")
run_audio("user2.wav")
