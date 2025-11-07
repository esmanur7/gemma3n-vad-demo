'''
OUTPUT

Ses işleniyor: user1.wav

/usr/local/lib/python3.12/dist-packages/torchaudio/_backend/utils.py:213: UserWarning: In 2.9, this function's implementation will be changed to use torchaudio.load_with_torchcodec` under the hood. Some parameters like ``normalize``, ``format``, ``buffer_size``, and ``backend`` will be ignored. We recommend that you port your code to rely directly on TorchCodec's decoder instead: https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.AudioDecoder.html#torchcodec.decoders.AudioDecoder.
  warnings.warn(
/usr/local/lib/python3.12/dist-packages/torchaudio/_backend/ffmpeg.py:88: UserWarning: torio.io._streaming_media_decoder.StreamingMediaDecoder has been deprecated. This deprecation is part of a large refactoring effort to transition TorchAudio into a maintenance phase. The decoding and encoding capabilities of PyTorch for both audio and video are being consolidated into TorchCodec. Please see https://github.com/pytorch/audio/issues/3902 for more information. It will be removed from the 2.9 release. 
  s = torchaudio.io.StreamReader(src, format, None, buffer_size)

Model..

Asistan :user




The speech you hear is the user's message. Respond as if you are talking to them directly.Just answer naturally like in a conversation.Respond in Turkish.
model
Evet, bu çok ilginç bir soru! Robotların gelecekte ev işlerini yapıp yapamayacağını merak ediyor musun? Eğer yapabilirlerse, hangi alanlarda? Belki mutfakta yemek hazırlama, çamaşır yıkama, temizlik gibi işler yapabilirler. Ya da belki daha karmaşık işler, örneğin bahçe işleri, çocuk bakımı gibi alanlarda yardımcı olabilirler.

Şu anda gördüğümüz robotlar genellikle insanı taklit eden, belirli görevleri yerine getiren robotlar. Ama gelecekte daha çok farklı türlerde robotlar çıkabilir. Örneğin, daha küçük ve esnek robotlar olabilirler, bunlar eşyaları taşıma, ince detayları işleme gibi görevlerde kullanılabilir. Ya da belki daha büyük ve güçlü robotlar olabilirler, bunlar inşaat işlerinde, madencilikte veya keşif görevlerinde kullanılabilir. 

Senin aklında hangi robot türleri var?


Ses işleniyor: user2.wav
Model..

Asistan :user





model
user




The speech you hear is the user's message. Respond as if you are talking to them directly.Just answer naturally like in a conversation.Respond in Turkish.
model
Evet, bu çok ilginç bir soru! Robotların gelecekte ev işlerini yapıp yapamayacağını merak ediyor musun? Eğer yapabilirlerse, hangi alanlarda? Belki mutfakta yemek hazırlama, çamaşır yıkama, temizlik gibi işler yapabilirler. Ya da belki daha karmaşık işler, örneğin bahçe işleri, çocuk bakımı gibi alanlarda yardımcı olabilirler.

Şu anda gördüğümüz robotlar genellikle insanı taklit eden, belirli görevleri yerine getiren robotlar. Ama gelecekte daha çok farklı türlerde robotlar çıkabilir. Örneğin, daha küçük ve esnek robotlar olabilirler, bunlar eşyaları taşıma, ince detayları işleme gibi görevlerde kullanılabilir. Ya da belki daha büyük ve güçlü robotlar olabilirler, bunlar inşaat işlerinde, madencilikte veya keşif görevlerinde kullanılabilir. 

Senin aklında hangi robot türleri var?
user




The speech you hear is the user's message. Respond as if you are talking to them directly.Just answer naturally like in a conversation.Respond in Turkish.
model
Evet, eğitim çok önemli bir konu. İnsan hayatının neredeyse yarısından fazlasını eğitimle geçiriyoruz ve bu süreçte farklı aşamalarda farklı şeyler öğreniyoruz.

İlkokulda temel bilgi ve beceriler öğreniyoruz. Okuma, yazma, matematik gibi konuları öğreniyoruz ve aynı zamanda sosyal ve duygusal gelişimimizi de sağlıyoruz. Ortaokulda ise daha derinlemesine konulara geçiyoruz. Fen bilimleri, sosyal bilimler, tarih gibi dersleri öğreniyor ve daha karmaşık düşünme becerileri geliştirmeye başlıyoruz. Lisede ise üniversiteye hazırlık yapıyoruz. Daha çok teorik bilgiye odaklanıyor ve gelecekteki kariyerimiz için gerekli olan becerileri kazanmaya çalışıyoruz.

Üniversitede ise uzmanlaşma alanımızı seçiyoruz ve o alandaki bilgileri derinlemesine öğreniyoruz. Aynı zamanda araştırmalar yapıyoruz ve farklı disiplinlerle etkileşim kuruyoruz. Üniversite eğitimi, sadece bilgi edinmekle kalmıyor, aynı zamanda kişisel gelişimimizi de sağlıyor ve hayata hazırlıyor.

Eğitim sürecinde insanla robot arasındaki ilişki de önemli bir konu. Robotlar belirli görevleri yerine getirebilirler, ancak insan zekası, yaratıcılık ve empati gibi beceriler hala insanlara özgü. Eğitimde bu becerileri geliştirmeye odaklanmamız gerekiyor. Robotların bize yardımcı olabileceği alanları belirleyip, insanların daha çok yaratıcı ve problem çözme becerileri geliştirebileceği alanlara yönelmeliyoruz.

Eğitim sadece bilgi edinmekle kalmıyor, aynı zamanda sosyal ilişkilerimizi ve kişisel gelişimimizi de etkiliyor. Okulda arkadaşlıklar kuruyoruz, öğretmenlerle iletişim kuruyoruz ve farklı kültürleri tanıyoruz. Bu deneyimler sayesinde karakterimizi geliştiriyor ve gelecekteki hayatımız için hazırlık yapıyoruz.

Sonuç olarak, eğitim hayatımızın en önemli aşamalarından biri. Farklı eğitim seviyelerinde farklı şeyler öğreniyoruz ve bu süreçte insanla robot arasındaki ilişkiyi de göz önünde bulundurarak geleceğe hazırlanmalıyız.


'''

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


