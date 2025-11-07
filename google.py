import torch, torchaudio
from transformers import AutoProcessor, AutoModelForImageTextToText

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

vad_model, utils = torch.hub.load("snakers4/silero-vad", "silero_vad", trust_repo=True)
(get_speech_timestamps, _, _, _, collect_chunks) = utils

def apply_vad(wav, sr):
   
    if wav.ndim == 2:
        wav = wav.mean(dim=0, keepdim=True)  # (1, N)

    speech_timestamps = get_speech_timestamps(wav, vad_model, sampling_rate=sr)

    if len(speech_timestamps) == 0:
        return None

    chunks = []
    for ts in speech_timestamps:
        start, end = int(ts["start"]), int(ts["end"])
        chunks.append(wav[:, start:end])

    wav = torch.cat(chunks, dim=1)  
    return wav.squeeze(0) 

MODEL_ID = "google/gemma-3n-E2B-it"
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
).eval()

history = []   

def run_audio(path):
    global history

    print(f"\nSes işleniyor: {path}")

    wav, sr = torchaudio.load(path)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000

    wav = apply_vad(wav, sr)
    if wav is None:
        print("Konuşma algılanmadı.\n")
        return

    wav = wav.cpu().numpy()

    messages = history + [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav},
                {"type": "text", "text": "The speech you hear is the user's message. Respond as if you are talking to them directly.Just answer naturally like in a conversation.Respond in Turkish."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    print("Model düşünüyor..\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.7,
            top_p=0.9
        )

    reply = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(f"Asistan :{reply}\n")

    history.append({"role": "user", "content": [{"type": "audio", "audio": wav}]})
    history.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})

run_audio("user1.wav")
run_audio("user2.wav")


