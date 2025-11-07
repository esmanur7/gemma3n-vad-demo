'''
!pip install -q "torch>=2.4.0" "transformers>=4.53.0" torchaudio soundfile accelerate bitsandbytes

from huggingface_hub import login
login()

from google.colab import files
uploaded = files.upload()

!apt -y install ffmpeg
!ffmpeg -y -i "WhatsApp Audio 2025-11-07 at 10.12.48.mp4" -ac 1 -ar 16000 user1.wav
!ffmpeg -y -i "WhatsApp Audio 2025-11-07 at 10.12.49.mp4" -ac 1 -ar 16000 user2.wav
'''

import torch, torchaudio
from transformers import AutoProcessor, AutoModelForImageTextToText

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
    print(f"\n Ses işleniyor :{path}")

    wav, sr = torchaudio.load(path)

    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0).cpu().numpy()   

    messages = history + [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav},
                {"type": "text", "text": "The speech you hear is the user's message. Respond as if you are talking to them directly.Just answer naturally like in a conversation."}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = inputs.to(model.device, dtype=model.dtype)

    print("Model..\n")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.9,
        )

    reply = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print(f"Asistan :{reply}\n")

    history.append({"role": "user", "content": [{"type": "audio", "audio": wav}]})
    history.append({"role": "assistant", "content": [{"type": "text", "text": reply}]})


run_audio("user1.wav")
run_audio("user2.wav")
