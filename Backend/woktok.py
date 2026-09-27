"""
WokTok CLI - Offline Multilingual Speech Engine
Type English or speak English -> hear it in 10 Indian languages.
"""

import os
import uuid
import warnings
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf

# FFmpeg injection (before transformers)
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

TRANSLATION_MODEL_PATH = os.path.join(MODELS_DIR, "indictrans2")
TTS_MODEL_PATH = os.path.join(MODELS_DIR, "indic-parler-tts")
WHISPER_MODEL_PATH = os.path.join(MODELS_DIR, "whisper-small")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LANGUAGES (10)
# ============================================================
LANGUAGES = {
    "1":  {"name": "Hindi",     "code": "hin_Deva", "scheme": sanscript.DEVANAGARI},
    "2":  {"name": "Gujarati",  "code": "guj_Gujr", "scheme": sanscript.GUJARATI},
    "3":  {"name": "Marathi",   "code": "mar_Deva", "scheme": sanscript.DEVANAGARI},
    "4":  {"name": "Telugu",    "code": "tel_Telu", "scheme": sanscript.TELUGU},
    "5":  {"name": "Punjabi",   "code": "pan_Guru", "scheme": sanscript.GURMUKHI},
    "6":  {"name": "Tamil",     "code": "tam_Taml", "scheme": sanscript.TAMIL},
    "7":  {"name": "Assamese",  "code": "asm_Beng", "scheme": sanscript.BENGALI},
    "8":  {"name": "Bengali",   "code": "ben_Beng", "scheme": sanscript.BENGALI},
    "9":  {"name": "Kannada",   "code": "kan_Knda", "scheme": sanscript.KANNADA},
    "10": {"name": "Malayalam", "code": "mal_Mlym", "scheme": sanscript.MALAYALAM},
}

VOICE_PRESETS = {
    "1": {"label": "Default (Female, Moderate, Clear)",
          "desc": "A female speaker delivers a clear and articulate reading with a moderate pace and quiet background."},
    "2": {"label": "Expressive (Male, Enthusiastic)",
          "desc": "An enthusiastic male speaker reads in a very expressive and animated tone."},
    "3": {"label": "Calm (Female, Soft)",
          "desc": "A soft-spoken female speaker delivers a calm, gentle reading in a close-up recording."},
    "4": {"label": "Energetic (Female, Fast)",
          "desc": "A young female speaker delivers an energetic reading at a fast pace with very clear audio."},
    "5": {"label": "Authoritative (Male, Deep, Slow)",
          "desc": "A male speaker with a deep pitch speaks slowly and clearly in a quiet environment."},
}

# ============================================================
# LOAD MODELS
# ============================================================
print("=" * 70)
print("WokTok - Loading Models")
print("=" * 70)

print("[1/3] Loading IndicTrans2 (English -> Indic)...")
translation_tokenizer = AutoTokenizer.from_pretrained(
    TRANSLATION_MODEL_PATH, trust_remote_code=True, local_files_only=True
)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(
    TRANSLATION_MODEL_PATH, trust_remote_code=True, local_files_only=True
).to(DEVICE)
translation_model.eval()

print("[2/3] Loading Indic Parler-TTS...")
tts_device = "cuda:0" if torch.cuda.is_available() else "cpu"
tts_model = ParlerTTSForConditionalGeneration.from_pretrained(
    TTS_MODEL_PATH, local_files_only=True
).to(tts_device)
tts_model.eval()
tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_PATH, local_files_only=True)
description_tokenizer = AutoTokenizer.from_pretrained(
    tts_model.config.text_encoder._name_or_path, local_files_only=True
)

print("[3/3] Loading Whisper (English speech-to-text)...")
whisper_pipe = pipeline(
    "automatic-speech-recognition",
    model=WHISPER_MODEL_PATH,
    device=-1,
    model_kwargs={"local_files_only": True}
)
whisper_pipe.model.config.forced_decoder_ids = None
if hasattr(whisper_pipe.model, "generation_config"):
    whisper_pipe.model.generation_config.forced_decoder_ids = None

print("\n[SUCCESS] All models ready.")
print("=" * 70)

# ============================================================
# CORE FUNCTIONS
# ============================================================

def record_voice(duration=5, sample_rate=16000):
    print(f"\nRecording for {duration} seconds... Speak now.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="float32")
    sd.wait()
    print("Recording complete. Transcribing...")
    result = whisper_pipe(
        {"raw": audio.squeeze().astype(np.float32), "sampling_rate": sample_rate},
        generate_kwargs={"language": "english", "task": "transcribe"}
    )
    text = result.get("text", "").strip()
    print(f"Transcribed: \"{text}\"")
    return text


def translate(text, lang_info):
    formatted = f"eng_Latn {lang_info['code']} {text}"
    inputs = translation_tokenizer(
        formatted, return_tensors="pt", padding=True, truncation=True
    ).to(DEVICE)
    with torch.no_grad():
        tokens = translation_model.generate(**inputs, max_length=256, num_beams=5)
    raw = translation_tokenizer.batch_decode(tokens, skip_special_tokens=True)[0].strip()
    scheme = lang_info.get("scheme")
    if scheme and scheme != sanscript.DEVANAGARI:
        native = transliterate(raw, sanscript.DEVANAGARI, scheme)
    else:
        native = raw
    return native.strip()


def synthesize(text, lang_name, voice_desc):
    desc_inputs = description_tokenizer(voice_desc, return_tensors="pt").to(tts_device)
    prompt_inputs = tts_tokenizer(text, return_tensors="pt").to(tts_device)
    with torch.no_grad():
        generation = tts_model.generate(
            input_ids=desc_inputs.input_ids,
            attention_mask=desc_inputs.attention_mask,
            prompt_input_ids=prompt_inputs.input_ids,
            prompt_attention_mask=prompt_inputs.attention_mask,
        )
    audio_arr = generation.cpu().numpy().squeeze()
    filename = f"woktok_{lang_name.lower()}_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    sf.write(filepath, audio_arr, tts_model.config.sampling_rate)
    return filepath


def play_audio(filepath):
    print("Playing...")
    data, sr = sf.read(filepath)
    sd.play(data, sr)
    sd.wait()
    print("Done.")


# ============================================================
# INTERACTIVE
# ============================================================

def select_language():
    print("\n" + "-" * 50)
    print("Select Target Language:")
    for key, val in LANGUAGES.items():
        print(f"  [{key:>2}] {val['name']}")
    print("-" * 50)
    choice = input("Enter number (default 1 = Hindi): ").strip()
    selected = LANGUAGES.get(choice, LANGUAGES["1"])
    print(f"-> Language: {selected['name']} ({selected['code']})")
    return selected


def select_voice():
    print("\n" + "-" * 50)
    print("Select Voice Style:")
    for key, val in VOICE_PRESETS.items():
        print(f"  [{key}] {val['label']}")
    print("  [C] Custom description")
    print("-" * 50)
    choice = input("Enter choice (default 1): ").strip().upper()
    if choice == "C":
        custom = input("Describe the voice: ").strip()
        if custom:
            return custom
    preset = VOICE_PRESETS.get(choice, VOICE_PRESETS["1"])
    print(f"-> Voice: {preset['label']}")
    return preset["desc"]


def run_once(english_text, lang_info, voice_desc, run_num):
    print(f"\n{'=' * 60}")
    print(f"  RUN #{run_num}  |  {lang_info['name']}")
    print(f"{'=' * 60}")
    print(f"  English: {english_text}")
    native_text = translate(english_text, lang_info)
    print(f"  {lang_info['name']}: {native_text}")
    print(f"  Generating audio...")
    audio_path = synthesize(native_text, lang_info["name"], voice_desc)
    print(f"  Saved: {audio_path}")
    play_audio(audio_path)
    print(f"{'=' * 60}\n")


def main():
    print("\n" + "#" * 60)
    print("  WokTok - Offline Multilingual Speech Engine")
    print("  English -> 10 Indian Languages")
    print("#" * 60)

    lang_info = select_language()
    voice_desc = select_voice()

    print("\n" + "=" * 60)
    print("  Commands:")
    print("    <text>       Type English text -> translate + speak")
    print("    v or voice   Record voice -> transcribe + translate + speak")
    print("    lang         Change target language")
    print("    voice-style  Change voice style")
    print("    exit / quit  Exit")
    print("=" * 60)

    run_count = 0
    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            cmd = user_input.lower()

            if cmd in ("exit", "quit"):
                print("Goodbye.")
                break
            if cmd == "lang":
                lang_info = select_language()
                continue
            if cmd in ("voice-style", "style"):
                voice_desc = select_voice()
                continue
            if cmd in ("v", "voice"):
                english_text = record_voice(duration=5)
                if not english_text:
                    print("No speech detected.")
                    continue
            else:
                english_text = user_input

            run_count += 1
            run_once(english_text, lang_info, voice_desc, run_count)

        except KeyboardInterrupt:
            print("\n\nSession ended.")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()