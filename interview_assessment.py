import streamlit as st
import requests
import re
import tempfile
import subprocess
import os

# ========================= CONFIG =========================
st.set_page_config(page_title="AI Interview Assessment", layout="wide")

HF_TOKEN = st.secrets["HF_TOKEN"]

# Endpoint HF API (Ganti sesuai endpoint kamu)
HF_WHISPER_API = "https://api-inference.huggingface.co/models/openai/whisper-tiny"
HF_PHI3_API = "https://api-inference.huggingface.co/models/microsoft/Phi-3.5-mini-instruct"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

INTERVIEW_QUESTIONS = [
    "Can you share any specific challenges you faced while working on certification and how you overcame them?",
    "Can you describe your experience with transfer learning in TensorFlow? How did it benefit your projects?",
    "Describe a complex TensorFlow model you have built and the steps you took to ensure its accuracy and efficiency.",
    "Explain how to implement dropout in a TensorFlow model and the effect it has on training.",
    "Describe the process of building a convolutional neural network (CNN) using TensorFlow for image classification."
]

CRITERIA = (
    "Kriteria Penilaian:\n"
    "0 - not answer the question\n"
    "1 - the answer is not relevan for question\n"
    "2 - Understand for general question\n"
    "3 - Understand with practice solution\n"
    "4 - Deep understanding with inovative solution\n"
)

# ========================= HF API CALLS =========================
def hf_asr_api(audio_bytes):
    """Kirim audio ke HF Whisper API"""
    try:
        response = requests.post(
            HF_WHISPER_API,
            headers=HEADERS,
            data=audio_bytes
        )
        result = response.json()
        return result.get("text", "TRANSCRIBE ERROR: No text returned")
    except Exception as e:
        return f"TRANSCRIBE ERROR: {e}"


def hf_llm_api(prompt):
    """Kirim prompt ke Phi-3 API"""
    try:
        data = {"inputs": prompt, "parameters": {"max_new_tokens": 200}}
        response = requests.post(HF_PHI3_API, headers=HEADERS, json=data)
        result = response.json()

        if isinstance(result, list):
            return result[0].get("generated_text", "")
        if "generated_text" in result:
            return result["generated_text"]
        return str(result)
    except Exception as e:
        return f"LLM ERROR: {e}"


# ========================= FUNCTIONS =========================
def convert_video_to_audio(video_bytes):
    """Convert MP4 → WAV 16kHz mono via ffmpeg"""
    try:
        subprocess.run(["ffmpeg", "-version"], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None, "FFMPEG NOT INSTALLED"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
        tmp_in.write(video_bytes)
        tmp_in_path = tmp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_out:
        tmp_out_path = tmp_out.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_in_path, "-ac", "1", "-ar", "16000", tmp_out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
    except Exception as e:
        return None, f"FFMPEG ERROR: {e}"

    with open(tmp_out_path, "rb") as f:
        audio_bytes = f.read()

    os.remove(tmp_in_path)
    os.remove(tmp_out_path)

    return audio_bytes, None


def prompt_for_classification(question, answer):
    return (
        "You are an expert HR interviewer and technical evaluator. Your task is to objectively assess the "
        "candidate's response based solely on the provided transcript. You must classify the answer using a strict "
        "0 until 4 scoring rubric.\n\n"
        f"{CRITERIA}\n\n"
        "Evaluation Rules:\n"
        "- Evaluate ONLY based on the candidate's answer.\n"
        "- Do NOT add missing information, assumptions, or corrections.\n"
        "- Judge relevance, accuracy, clarity, and depth based on the rubric.\n"
        "- Your explanation must be concise and directly tied to the rubric.\n"
        "- You MUST follow the output format exactly.\n\n"
        f"Question:\n{question}\n\n"
        f"Candidate Answer (Transcript):\n{answer}\n\n"
        "Required Output Format:\n"
        "KLASIFIKASI: <angka>\n"
        "ALASAN: <teks>\n"
    )


def parse_model_output(text):
    score_match = re.search(r"KLASIFIKASI[:\- ]*([0-4])", text, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else None

    reason_match = re.search(r"ALASAN[:\- ]*(.+)", text, re.IGNORECASE | re.DOTALL)
    reason = reason_match.group(1).strip() if reason_match else text

    return score, reason


# ========================= SESSION STATE =========================
for key, default in {
    "page": "input",
    "results": [],
    "nama": "",
    "processing_done": False
}.items():
    st.session_state.setdefault(key, default)


# ========================= PAGE INPUT =========================
if st.session_state.page == "input":
    st.title("🎥 AI-Powered Interview Assessment System")
    st.write("Upload **5 video interview** lalu klik mulai analisis.")

    with st.form("upload_form"):
        nama = st.text_input("Nama Pelamar")
        uploaded = st.file_uploader(
            "Upload 5 Video (1 → 5)",
            type=["mp4", "mov", "mkv", "webm"],
            accept_multiple_files=True
        )
        submit = st.form_submit_button("Mulai Proses Analisis")

    if submit:
        if not nama:
            st.error("Nama wajib diisi.")
        elif not uploaded or len(uploaded) != 5:
            st.error("Harap upload tepat 5 video.")
        else:
            st.session_state.nama = nama
            st.session_state.uploaded = uploaded
            st.session_state.results = []
            st.session_state.page = "result"
            st.session_state.processing_done = True
            st.rerun()


# ========================= PAGE RESULT =========================
if st.session_state.page == "result":
    st.title("📋 Hasil Penilaian Interview")
    st.write(f"**Nama Pelamar:** {st.session_state.nama}")

    progress = st.empty()

    if len(st.session_state.results) == 0:
        for idx, vid in enumerate(st.session_state.uploaded):
            progress.info(f"Memproses Video {idx+1}...")

            # Convert ke audio
            audio_bytes, err = convert_video_to_audio(vid.read())
            if err:
                st.session_state.results.append({
                    "question": INTERVIEW_QUESTIONS[idx],
                    "transcript": err,
                    "score": None,
                    "reason": err,
                    "raw_model": err
                })
                continue

            # Transcribe
            transcript = hf_asr_api(audio_bytes)

            # LLM Classification
            prompt = prompt_for_classification(INTERVIEW_QUESTIONS[idx], transcript)
            raw_output = hf_llm_api(prompt)
            score, reason = parse_model_output(raw_output)

            st.session_state.results.append({
                "question": INTERVIEW_QUESTIONS[idx],
                "transcript": transcript,
                "score": score,
                "reason": reason,
                "raw_model": raw_output
            })

            progress.success(f"Video {idx+1} selesai ✔")

    # Final Score
    scores = [r["score"] for r in st.session_state.results if r["score"] is not None]
    if len(scores) == 5:
        final_score = sum(scores) / 5
        st.markdown(f"### ⭐ Skor Akhir: **{final_score:.2f} / 4**")
    else:
        st.error("Tidak semua skor berhasil diproses.")

    st.markdown("---")

    # Detail per video
    for i, r in enumerate(st.session_state.results):
        st.subheader(f"🎬 Video {i+1}")
        st.write(f"**Pertanyaan:** {r['question']}")
        st.write(f"**Transkrip:** {r['transcript']}")
        st.write(f"**Skor:** {r['score']}")
        st.write(f"**Alasan:** {r['reason']}")

        with st.expander("Raw Output Model"):
            st.code(r["raw_model"])

        st.markdown("---")

    if st.button("🔙 Kembali"):
        st.session_state.page = "input"
        st.session_state.processing_done = False
        st.session_state.results = []
        st.session_state.nama = ""
        st.rerun()
