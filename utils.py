import whisper
import subprocess
import os


model= whisper.load_model("base")

def transcribe_audio(audio_path):
    result=model.transcribe(audio_path)
    return result

def extract_audio_from_video(video_path):
    base_name = os.path.splitext(video_path)[0]
    audio_path = base_name + ".wav"

    command = [
    "ffmpeg",
    "-y",                  # overwrite
    "-i", video_path,      # input
    "-vn",                 # remove video
    "-acodec", "libmp3lame",
    audio_path
    ]
    subprocess.run(command,stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

#OCR
from pdf2image import convert_from_path
import pytesseract
from langchain_core.documents import Document

def load_scanned_pdf(file_path):
    images = convert_from_path(file_path)

    documents = []
    for i,img in enumerate(images):
        text=pytesseract.image_to_string(img)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source":"scanned_pdf",
                    "page":i
                }
            )
        )
    return documents