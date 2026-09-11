import streamlit as st
import uuid


from ingest import load_pdf, load_audio, load_video
from rag import create_vectorstore, create_rag_chain

st.set_page_config(page_title="Multimodal RAG", layout="wide")

st.title("🚀 Multimodal RAG (PDF + Audio + Video + OCR)")


# session state

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "docs" not in st.session_state:
    st.session_state.docs = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None   # FIXED TYPO

if "rag" not in st.session_state:
    st.session_state.rag = None

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()



# Upload


files = st.file_uploader(
    "Upload Files",
    type=["pdf", "mp3", "wav", "mp4"],
    accept_multiple_files=True
)

if files:
    for file in files:

        # prevent duplicate processing
        if file.name in st.session_state.processed_files:
            continue

        path = f"temp_{file.name}"

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        docs = []

        if file.name.endswith(".pdf"):
            docs = load_pdf(path)

        elif file.name.endswith((".mp3", ".wav")):
            docs = load_audio(path)

        elif file.name.endswith(".mp4"):
            docs = load_video(path)

        st.session_state.docs.extend(docs)
        st.session_state.processed_files.add(file.name)

        st.success(f"Processed {file.name}")


# Build Index

if st.button("Create Knowledge Base"):

    if len(st.session_state.docs) == 0:
        st.warning("No documents uploaded!")
    else:
        with st.spinner("Building vector database..."):

            st.session_state.vectorstore = create_vectorstore(st.session_state.docs)
            st.session_state.rag = create_rag_chain(st.session_state.vectorstore)

        st.success("✅ Knowledge Base Ready!")


# Query

q = st.text_input("Ask Question")

if q:

    if st.session_state.rag is None:
        st.warning("Please create knowledge base first.")
    else:
        with st.spinner("Thinking..."):

            response = st.session_state.rag.invoke(
                {"question": str(q)},
                config={
                    "configurable": {
                        "session_id": str(st.session_state.session_id)
                    }
                }
            )

        st.subheader("Answer")
        st.write(response.content)