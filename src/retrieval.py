# src/retrieval.py
from sentence_transformers import SentenceTransformer
import faiss
import dropbox
import os
import pickle
import logging

from config import DROPBOX_TOKEN, DROPBOX_FOLDER, FAISS_INDEX_PATH

# Configurer le logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        self.dbx = dropbox.Dropbox(DROPBOX_TOKEN)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        if os.path.exists(FAISS_INDEX_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_INDEX_PATH + '.docs', 'rb') as f:
                self.documents = pickle.load(f)
            logger.info("Index FAISS chargé depuis le fichier existant.")
        else:
            self.documents = self.download_documents()
            if self.documents:
                self.index = self.build_index(self.documents)
                logger.info("Index FAISS créé avec succès.")
            else:
                logger.warning("Aucun document trouvé. L'index FAISS ne sera pas créé.")
                self.index = None

    def download_documents(self):
        documents = []
        try:
            res = self.dbx.files_list_folder(DROPBOX_FOLDER)
            for entry in res.entries:
                if isinstance(entry, dropbox.files.FileMetadata):
                    metadata, res_file = self.dbx.files_download(entry.path_lower)
                    documents.append(res_file.content.decode('utf-8'))
            logger.info(f"{len(documents)} documents téléchargés depuis Dropbox.")
        except dropbox.exceptions.ApiError as e:
            logger.error(f"Erreur lors du téléchargement des documents: {e}")
        return documents

    def build_index(self, documents):
        vectors = self.model.encode(documents, convert_to_numpy=True)
        if vectors.size == 0:
            logger.error("Aucun vecteur généré. Vérifiez vos documents.")
            raise ValueError("Aucun vecteur généré. Vérifiez vos documents.")
        dimension = vectors.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_INDEX_PATH + '.docs', 'wb') as f:
            pickle.dump(documents, f)
        logger.info("Index FAISS enregistré avec succès.")
        return index

    def retrieve(self, query, top_k=5):
        if not self.index:
            logger.warning("L'index FAISS n'est pas disponible.")
            return []
        query_vector = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vector, top_k)
        logger.info(f"Recherche FAISS exécutée pour la requête: {query}")
        return [self.documents[i] for i in I[0]]
