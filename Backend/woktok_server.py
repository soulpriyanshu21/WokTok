"""
WokTok Backend Server
Conversation translator (any Indian language <-> any Indian language)
"""

import os
import uuid
import tempfile
import subprocess
import warnings
import torch
import numpy as np
import soundfile as sf

try:
    import imageio_ffmpeg
    ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_bin)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    target_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if not os.path.exists(target_exe) and os.path.exists(ffmpeg_bin):
        import shutil
        shutil.copyfile(ffmpeg_bin, target_exe)
    print(f"[FFmpeg] {target_exe}")
except Exception as e:
    print(f"[FFmpeg Warning] {e}")

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from parler_tts import ParlerTTSForConditionalGeneration
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = r"C:\Users\ASUS\OneDrive\Desktop\WokTok\Backend"
MODELS_DIR = os.path.join(BASE_DIR, "models")

EN_INDIC_MODEL_PATH = os.path.join(MODELS_DIR, "indictrans2")
INDIC_INDIC_MODEL_PATH = os.path.join(MODELS_DIR, "indictrans2-indic-indic")
TTS_MODEL_PATH = os.path.join(MODELS_DIR, "indic-parler-tts")
WHISPER_MODEL_PATH = os.path.join(MODELS_DIR, "whisper-small")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
QUEUE_DIR = os.path.join(OUTPUT_DIR, "queue")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(QUEUE_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LANGUAGES
# ============================================================

LANGUAGES = {
    "english":   {"name": "English",   "code": "eng_Latn", "whisper_lang": "english",   "scheme": sanscript.DEVANAGARI},
    "hindi":     {"name": "Hindi",     "code": "hin_Deva", "whisper_lang": "hindi",     "scheme": sanscript.DEVANAGARI},
    "gujarati":  {"name": "Gujarati",  "code": "guj_Gujr", "whisper_lang": "gujarati",  "scheme": sanscript.GUJARATI},
    "marathi":   {"name": "Marathi",   "code": "mar_Deva", "whisper_lang": "marathi",   "scheme": sanscript.DEVANAGARI},
    "telugu":    {"name": "Telugu",    "code": "tel_Telu", "whisper_lang": "telugu",    "scheme": sanscript.TELUGU},
    "punjabi":   {"name": "Punjabi",   "code": "pan_Guru", "whisper_lang": "punjabi",   "scheme": sanscript.GURMUKHI},
    "tamil":     {"name": "Tamil",     "code": "tam_Taml", "whisper_lang": "tamil",     "scheme": sanscript.TAMIL},
    "assamese":  {"name": "Assamese",  "code": "asm_Beng", "whisper_lang": "assamese",  "scheme": sanscript.BENGALI},
    "bengali":   {"name": "Bengali",   "code": "ben_Beng", "whisper_lang": "bengali",   "scheme": sanscript.BENGALI},
    "kannada":   {"name": "Kannada",   "code": "kan_Knda", "whisper_lang": "kannada",   "scheme": sanscript.KANNADA},
    "malayalam": {"name": "Malayalam", "code": "mal_Mlym", "whisper_lang": "malayalam", "scheme": sanscript.MALAYALAM},
}

VOICE_DESC_DEFAULT = "A female speaker delivers a clear and articulate reading with a moderate pace and quiet background."

# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("WokTok Server - Loading Models")
print("=" * 70)

print("[1/4] Loading IndicTrans2 (English -> Indic)...")
en_indic_tokenizer = AutoTokenizer.from_pretrained(
    EN_INDIC_MODEL_PATH, trust_remote_code=True, local_files_only=True
)
en_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
    EN_INDIC_MODEL_PATH, trust_remote_code=True, local_files_only=True
).to(DEVICE)
en_indic_model.eval()

print("[2/4] Loading IndicTrans2 (Indic -> Indic)...")
indic_indic_tokenizer = AutoTokenizer.from_pretrained(
    INDIC_INDIC_MODEL_PATH, trust_remote_code=True, local_files_only=True
)
indic_indic_model = AutoModelForSeq2SeqLM.from_pretrained(
    INDIC_INDIC_MODEL_PATH, trust_remote_code=True, local_files_only=True
).to(DEVICE)
indic_indic_model.eval()

print("[3/4] Loading Parler-TTS...")
tts_device = "cuda:0" if torch.cuda.is_available() else "cpu"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
    TTS_MODEL_PATH, local_files_only=True
).to(tts_device)
tts_model.eval()
tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_PATH, local_files_only=True)
description_tokenizer = AutoTokenizer.from_pretrained(
    tts_model.config.text_encoder._name_or_path, local_files_only=True
)

print("[4/4] Loading Whisper...")
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL_PATH,
    device=-1,
    model_kwargs={"local_files_only": True}
)
whisper_pipe.model.config.forced_decoder_ids = None
if hasattr(whisper_pipe.model, "generation_config"):
    whisper_pipe.model.generation_config.forced_decoder_ids = None

print("\n[SUCCESS] All models loaded.")
print("=" * 70)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="WokTok API", version="1.0")

conversation_queue = []


class TTSRequest(BaseModel):
    text: str
    target_language: str = "hindi"


@app.get("/")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "languages": list(LANGUAGES.keys()),
    }


@app.post("/send")
async def send_message(
    file: UploadFile = File(...),
    source_lang: str = Form(...),
    target_lang: str = Form(...),
):
    if source_lang not in LANGUAGES:
        raise HTTPException(400, f"Unknown source language: {source_lang}")
    if target_lang not in LANGUAGES:
        raise HTTPException(400, f"Unknown target language: {target_lang}")

    try:
        suffix = ".m4a" if file.filename.lower().endswith(".m4a") else ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        wav_path = tmp_path.rsplit(".", 1)[0] + "_16k.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
            check=True, capture_output=True
        )

        # 1. Whisper
        audio_data, sr = sf.read(wav_path)
        whisper_lang = LANGUAGES[source_lang]["whisper_lang"]
        asr_result = whisper_pipe(
            {"raw": audio_data.astype(np.float32), "sampling_rate": sr},
            generate_kwargs={"language": whisper_lang, "task": "transcribe"}
        )
        source_text = asr_result["text"].strip()
        if not source_text:
            return {"status": "empty", "message": "No speech detected"}

        # 2. Translate source -> target
        src_code = LANGUAGES[source_lang]["code"]
        tgt_code = LANGUAGES[target_lang]["code"]

        if source_lang == target_lang:
            native_text = source_text
        else:
            formatted = f"{src_code} {tgt_code} {source_text}"
            inputs = indic_indic_tokenizer(
                formatted, return_tensors="pt", padding=True, truncation=True
            ).to(DEVICE)
            with torch.no_grad():
                tokens = indic_indic_model.generate(**inputs, max_length=256, num_beams=5)
            raw = indic_indic_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()

            tgt_scheme = LANGUAGES[target_lang]["scheme"]
            if tgt_scheme != sanscript.DEVANAGARI:
                native_text = transliterate(raw, sanscript.DEVANAGARI, tgt_scheme)
            else:
                native_text = raw

        # 3. TTS
        desc_inputs = description_tokenizer(VOICE_DESC_DEFAULT, return_tensors="pt").to(tts_device)
        prompt_inputs = tts_tokenizer(native_text, return_tensors="pt").to(tts_device)
        with torch.no_grad():
            generation = tts_model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
        audio_arr = generation.cpu().numpy().squeeze()

        msg_id = uuid.uuid4().hex[:12]
        audio_file = os.path.join(QUEUE_DIR, f"{msg_id}.wav")
        sf.write(audio_file, audio_arr, tts_model.config.sampling_rate)

        conversation_queue.append({
            "id": msg_id,
            "audio_path": audio_file,
            "source_text": source_text,
            "translated_text": native_text,
            "from_lang": source_lang,
            "to_lang": target_lang,
        })

        try:
            os.unlink(tmp_path)
            os.unlink(wav_path)
        except Exception:
            pass

        print(f"[SEND] {source_lang} -> {target_lang}: '{source_text}' -> '{native_text}'")
        return {
            "status": "ok",
            "id": msg_id,
            "source_text": source_text,
            "translated_text": native_text,
        }

    except Exception as e:
        print(f"[SEND ERROR] {e}")
        raise HTTPException(500, str(e))


@app.get("/receive")
def receive_message():
    if not conversation_queue:
        return {"status": "empty"}
    msg = conversation_queue.pop(0)
    return {
        "status": "ok",
        "id": msg["id"],
        "audio_url": f"/audio/{msg['id']}",
        "source_text": msg["source_text"],
        "translated_text": msg["translated_text"],
        "from_lang": msg["from_lang"],
        "to_lang": msg["to_lang"],
    }


@app.get("/audio/{msg_id}")
def get_audio(msg_id: str):
    filepath = os.path.join(QUEUE_DIR, f"{msg_id}.wav")
    if not os.path.exists(filepath):
        raise HTTPException(404, "Audio not found")
    return FileResponse(filepath, media_type="audio/wav")


@app.post("/synthesize")
def synthesize(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(400, "Text is empty")
    lang_key = req.target_language.lower()
    if lang_key not in LANGUAGES:
        raise HTTPException(400, f"Unsupported: {lang_key}")

    lang_info = LANGUAGES[lang_key]
    try:
        formatted = f"eng_Latn {lang_info['code']} {req.text}"
        inputs = en_indic_tokenizer(
            formatted, return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        with torch.no_grad():
            tokens = en_indic_model.generate(**inputs, max_length=256, num_beams=5)
        raw = en_indic_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()

        if lang_info["scheme"] != sanscript.DEVANAGARI:
            native_text = transliterate(raw, sanscript.DEVANAGARI, lang_info["scheme"])
        else:
            native_text = raw

        desc_inputs = description_tokenizer(VOICE_DESC_DEFAULT, return_tensors="pt").to(tts_device)
        prompt_inputs = tts_tokenizer(native_text, return_tensors="pt").to(tts_device)
        with torch.no_grad():
            generation = tts_model.generate(
                input_ids=desc_inputs.input_ids,
                attention_mask=desc_inputs.attention_mask,
                prompt_input_ids=prompt_inputs.input_ids,
                prompt_attention_mask=prompt_inputs.attention_mask,
            )
        audio_arr = generation.cpu().numpy().squeeze()
        filename = f"woktok_{lang_key}_{uuid.uuid4().hex[:8]}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        sf.write(filepath, audio_arr, tts_model.config.sampling_rate)
        return FileResponse(filepath, media_type="audio/wav", filename=filename)
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)