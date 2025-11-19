import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# --- 1. CONFIGURACIÓN DE PARÁMETROS ---
DATA_PATH = "./data/knowledge.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" 
VECTOR_DB_COLLECTION = "consular_data"

# @st.cache_resource asegura que esta función se ejecute solo una vez al inicio del Space
@st.cache_resource
def setup_rag_system():
    # A. Cargar y segmentar documentos
    try:
        loader = TextLoader(DATA_PATH, encoding="utf-8")
        documents = loader.load()
    except FileNotFoundError:
        st.error(f"Error: El archivo de conocimiento no se encuentra en {DATA_PATH}.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # B. Crear Embeddings y Vector Store (ChromaDB en memoria)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = Chroma.from_documents(
        texts, 
        embeddings, 
        collection_name=VECTOR_DB_COLLECTION
    )
    
    # C. Configurar LLM (Gemini) y Cadena RAG
    # La clave GOOGLE_API_KEY se lee desde los secretos del Space
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    except Exception as e:
        # Esto atrapará errores si la clave no está configurada o es inválida
        st.error(f"Error al inicializar Gemini. Asegúrate de que la variable GOOGLE_API_KEY esté configurada correctamente.")
        return None
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), 
    )
    return qa_chain

# --- 2. INTERFAZ DE STREAMLIT ---

st.set_page_config(page_title="Asistente Consular")
st.title("🏛️ Asistente Consular Automatizado RAG")
st.caption("Responde preguntas sobre trámites basado en la base de conocimiento cargada.")

qa_chain = setup_rag_system()

if qa_chain:
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! ¿En qué trámite consular puedo ayudarte hoy?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.spinner("Consultando base de conocimiento y redactando..."):
            try:
                result = qa_chain.run(prompt) 
            except Exception as e:
                result = "Hubo un error al conectar con el modelo de IA. Por favor, revisa tu clave de API."
        
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)
