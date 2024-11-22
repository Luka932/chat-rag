# src/test_dropbox.py
import dropbox
from config import DROPBOX_TOKEN, DROPBOX_FOLDER

def list_files():
    dbx = dropbox.Dropbox(DROPBOX_TOKEN)
    try:
        res = dbx.files_list_folder(DROPBOX_FOLDER)
        print(f"Contenu du dossier '{DROPBOX_FOLDER}':")
        for entry in res.entries:
            print(f"- {entry.name}")
    except dropbox.exceptions.ApiError as e:
        print(f"Erreur lors de l'accès au dossier '{DROPBOX_FOLDER}': {e}")

if __name__ == "__main__":
    list_files()
