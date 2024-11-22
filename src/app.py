# src/app.py
import streamlit as st
from retrieval import Retriever
from generation import Generator

# Initialisation
retriever = Retriever()
generator = Generator()

st.title("Chatbot avec RAG")

user_input = st.text_input("Vous:", "")
temperature = st.slider("Température", 0.0, 1.0, 0.7)
use_rag = st.checkbox("Utiliser RAG", value=True)

if st.button("Envoyer"):
    if use_rag:
        retrieved_docs = retriever.retrieve(user_input)
        context = "\n".join(retrieved_docs)
        prompt = f"Contexte: {context}\nQuestion: {user_input}\nRéponse:"
    else:
        prompt = f"Question: {user_input}\nRéponse:"
    
    response = generator.generate(prompt, temperature)
    st.text_area("Chatbot:", value=response, height=200)
