from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
load_dotenv()



#embeddings
embedding=HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# In memory session store

store={}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id]=ChatMessageHistory()
    return store[session_id]

def create_vectorstore(documents):
    splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    docs = splitter.split_documents(documents)
    return FAISS.from_documents(docs,embedding)



# Metadata aware retriever

def get_filtered_retriever(vectorstore,source_filter=None):
    if source_filter:
        return vectorstore.as_retriever(
            search_kwargs={
                "k":5,
                "filter": {
                    "source":source_filter
                }
            }
        )
    return vectorstore.as_retriever(search_kwargs={"k":5})

def create_rag_chain(vectorstore):
    def format_docs(docs):
        formatted = []
        for d in docs:
            meta = d.metadata

            if meta.get("source") in ["audio","video"]:
                time_info = f"[{meta.get('start_time',0):.1f}s - {meta.get('end_time',0):.1f}s]"
            else:
                time_info=""
            formatted.append(
                f"{time_info}\n{d.page_content}"
            )
        return "\n\n".join(formatted)

    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer based only on the provided context. Also mention timestamps if available."),
        MessagesPlaceholder(variable_name="history"),
        ("human","Context:\n{context}\n\nQuestion: {question}")
    ])

    llm= ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0
    )

    

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    
    from operator import itemgetter

    chain = (
        {
            "context": itemgetter("question") | retriever | RunnableLambda(format_docs),
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        }
        | prompt
        | llm
    )

   

      # Add memory wrapper
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    return chain_with_memory

