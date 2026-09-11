# 🚀 Multimodal RAG (PDF + Audio + Video + OCR)

This project is a **Generative AI application using LangChain** that allows users to upload **PDF, audio, and video files** and ask questions over their content.

---

## 🔥 Features

* 📄 PDF ingestion (including scanned PDFs via OCR)
* 🎧 Audio transcription + querying
* 🎥 Video transcription with timestamp-based answers
* 🔍 Semantic search using FAISS vector database
* 💬 Chat-based querying with memory
* 🎯 Metadata filtering (PDF / audio / video)
* ⏱ Timestamp retrieval for audio/video responses
* 🌐 Simple Streamlit UI

---

## 🛠 Tech Stack

* LangChain (Runnables)
* FAISS (Vector Database)
* HuggingFace Embeddings
* Whisper (Audio/Video Transcription)
* PyMuPDF / OCR (Scanned PDFs)
* Streamlit (Frontend)
* Groq / LLM API

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Also install:

```bash
ffmpeg
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📂 How it Works

1. Upload files (PDF, audio, video)
2. System extracts text + metadata
3. Text is split and stored in FAISS
4. User asks a question
5. Relevant chunks are retrieved
6. LLM generates answer using context

---
