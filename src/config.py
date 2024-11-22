# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

DROPBOX_TOKEN = os.getenv('DROPBOX_TOKEN')
DROPBOX_FOLDER = '/documents_chat'  # Ajustez si nécessaire
FAISS_INDEX_PATH = '../models/faiss_index.faiss'
