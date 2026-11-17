# Multilingual Media RAG Assistant

An end-to-end audio and video intelligence pipeline built with **Python**, **LangChain (LCEL)**, **OpenAI Whisper**, **Sarvam AI**, **Mistral AI**, and **ChromaDB**. It ingests YouTube URLs or local media files in **English and Hindi**, normalizes and segments audio, transcribes speech, extracts structured insights, and enables grounded conversational Q&A over long-form transcripts via a **Streamlit** web interface.

---

## Key Features

* **Universal Media Ingestion:** Accepts YouTube links (`yt-dlp`) as well as local audio/video files (`.mp4`, `.mp3`, `.wav`, `.m4a`).
* **Whisper-Optimized Audio Normalization:** Converts dual-channel audio to **16 kHz mono WAV** using `pydub` + `FFmpeg` and segments long recordings into **10-minute chunks** to prevent memory bottlenecks.
* **Dual-Engine Bilingual Speech-to-Text:**
  * **English:** Transcribed locally on CPU using **OpenAI Whisper** (`small` model).
  * **Hindi / Indic Audio:** Routed to **Sarvam AI (`saaras:v2.5`)** with automatic **25-second sub-chunking** to comply with API duration limits and preserve transcription accuracy.
* **Map-Reduce Summarization & Structured Extraction:** Uses **LangChain LCEL** (`RunnablePassthrough`, `RunnableLambda`) and **Mistral AI (`mistral-small-latest`)** to generate concise titles, bullet-point summaries, **Action Items** (task, owner, deadline), **Key Decisions**, and **Open Questions**.
* **Persistent RAG Q&A Engine:** Splits transcripts into overlapping chunks (`500` chars, `50` overlap), embeds them locally using Hugging Face's **`all-MiniLM-L6-v2`**, stores vectors in **ChromaDB**, and answers user queries strictly grounded in retrieved context.

---

## System Architecture

```text
YouTube URL / Local Media (.mp4, .mp3, .wav)
        │
        ▼
[1. Audio Processor] ──► yt-dlp + FFmpeg/pydub (16 kHz Mono WAV, 10-min chunks)
        │
        ▼
[2. Bilingual STT Router]
   ├── English ──► Local OpenAI Whisper (small)
   └── Hindi   ──► 25s Sub-chunking ──► Sarvam AI (saaras:v2.5)
        │
        ▼
   Unified Transcript
        ├──► [3. LCEL Map-Reduce Chain (Mistral AI)] ──► Title, Summary, Action Items, Decisions, Questions
        └──► [4. Vector Store (ChromaDB + all-MiniLM-L6-v2)] ──► Top-k Similarity Retrieval ──► Grounded RAG Chat
```

---

## Project Structure

```text
Multilingual-Media-RAG/
├── app.py                     # Streamlit web interface & live status dashboard
├── main.py                    # End-to-end CLI pipeline runner
├── core/
│   ├── transcriber.py         # Bilingual STT routing (Whisper + Sarvam AI)
│   ├── summarizer.py          # LCEL Map-Reduce summarization & title generation
│   ├── extractor.py           # Structured extraction (action items, decisions, questions)
│   ├── vector_store.py        # Text splitting, Hugging Face embeddings & ChromaDB persistence
│   └── rag_engine.py          # LCEL retrieval-augmented Q&A chain
├── utils/
│   └── audio_processor.py     # YouTube audio extraction, 16 kHz mono conversion & chunking
├── requirements.txt           # Project dependencies
└── .env.example               # Environment variable template
```

---

## Tech Stack

| Layer | Tools & Libraries |
| :--- | :--- |
| **Orchestration & LLM** | Python, LangChain (LCEL), Mistral AI (`mistral-small-latest`) |
| **Speech-to-Text** | OpenAI Whisper (local), Sarvam AI (`saaras:v2.5`) |
| **Audio Processing** | `yt-dlp`, `pydub`, `FFmpeg` |
| **Vector DB & Embeddings** | ChromaDB, Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`) |
| **Frontend UI** | Streamlit |

---

## Setup & Installation

### 1. Prerequisites
* **Python 3.10+**
* **FFmpeg** installed and available in your system `PATH`:
  * **Windows:** `winget install Gyan.FFmpeg`
  * **macOS:** `brew install ffmpeg`
  * **Linux (Ubuntu/Debian):** `sudo apt install ffmpeg`

### 2. Clone & Install Dependencies
```bash
git clone [https://github.com/Priyansh5733/Multilingual-Media-RAG.git](https://github.com/Priyansh5733/Multilingual-Media-RAG.git)
cd Multilingual-Media-RAG

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
MISTRAL_API_KEY=your_mistral_api_key_here
SARVAM_API_KEY=your_sarvam_api_key_here
WHISPER_MODEL=small
```

### 4. Run the Application
**Launch the Streamlit UI:**
```bash
streamlit run app.py
```

**Or run via CLI:**
```bash
python main.py
```

---

## Key Engineering Tradeoffs

* **16 kHz Mono Normalization:** Downmixing stereo audio to single-channel 16 kHz WAV cuts file size and inference latency while matching Whisper's native training sample rate.
* **Dual-Tier Chunking:** Audio is segmented at **10 minutes** for local Whisper memory safety, then dynamically sub-chunked into **25-second segments** when routed to Sarvam AI's REST endpoint.
* **Map-Reduce over Stuffing:** Long transcripts exceed single-pass prompt budgets; splitting into `3,000`-character windows with `200`-character overlap preserves cross-boundary context before final synthesis.
* **Local Embeddings + Persistent Vector Store:** Running `all-MiniLM-L6-v2` locally with on-disk ChromaDB keeps retrieval zero-cost and low-latency for iterative Q&A sessions.

---

## Roadmap

* [ ] Preserve segment-level `[MM:SS]` timestamps in ChromaDB metadata for clickable citation jumps.
* [ ] Add hybrid retrieval (BM25 + dense vector similarity) for exact proper-noun and entity recall.
* [ ] Export structured summaries and Q&A logs directly to Markdown and PDF.
