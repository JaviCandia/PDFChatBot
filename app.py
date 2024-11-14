import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

# Configuración de la página de la aplicación Streamlit
st.set_page_config('PDFChatBot')
st.header("Ask questions about your PDF")

# Campo para ingresar la API Key de OpenAI
OPENAI_API_KEY = st.text_input('OpenAI API Key', type='password')

# Campo para subir un archivo PDF
pdf_obj = st.file_uploader("Upload your PDF", type="pdf", on_change=st.cache_resource.clear)

@st.cache_resource
def create_embeddings(pdf):
    # Lee el PDF y extrae el texto
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Filtro de contenido: eliminar líneas en blanco 
    text = "\n".join([line for line in text.split("\n") if line.strip() != ""])

    # Divide el texto en chunks (trozos) más pequeños
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Tamaño del chunk
        chunk_overlap=100,  # Superposición entre chunks
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Crea los embeddings utilizando el modelo especificado de HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base

# Si se sube un archivo PDF
if pdf_obj:
    # Crea la base de conocimiento a partir del PDF
    knowledge_base = create_embeddings(pdf_obj)
    # Campo para ingresar la pregunta del usuario
    user_question = st.text_area("Ask something about your PDF:")

    if user_question:
        # Establece la API Key de OpenAI en el entorno
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        # Busca en la base de conocimiento los documentos más similares a la pregunta del usuario
        docs = knowledge_base.similarity_search(user_question, 3)
        # Configura el modelo de lenguaje de OpenAI
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        # Carga la cadena de preguntas y respuestas
        chain = load_qa_chain(llm, chain_type="stuff")
        # Ejecuta la cadena de QA con los documentos relevantes y la pregunta del usuario
        respuesta = chain.run(input_documents=docs, question=user_question)

        # Muestra la respuesta en la aplicación
        st.write(respuesta)
