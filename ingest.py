from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from utils import transcribe_audio,extract_audio_from_video,load_scanned_pdf



# PDF 
def load_pdf(file_path):
    try:
        loader=PyMuPDFLoader(file_path)
        docs = loader.load()

        total_text = sum(len(d.page_content) for d in docs)

        if total_text < 50:
            return load_scanned_pdf(file_path)

        for d in docs:
            d.metadata.update({"source": "pdf", "file_name": file_path})

        return docs

    except Exception:
        return load_scanned_pdf(file_path)

# Audio(with timestamps)

def load_audio(file_path):
    result=transcribe_audio(file_path)

    documents=[]
    for seg in result["segments"]:
        documents.append(
            Document(
                page_content = seg["text"],
                metadata={
                    "source":"audio",
                    "file_name":file_path,
                    "start_time":seg["start"],
                    "end_time":seg["end"]
                }
            )
        )
    return documents


# Video

def load_video(file_path):
    audio_path = extract_audio_from_video(file_path)
    docs = load_audio(audio_path)

    for d in docs:
        d.metadata["source"] = "video"
    return docs

