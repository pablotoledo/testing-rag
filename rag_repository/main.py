import os
from git import Repo

import shutil

# remove /repo directory if it exists using python
if os.path.exists("repo"):
    shutil.rmtree("repo")

# Clona el repositorio si es necesario
repo_path = os.getcwd() + "/repo"
# repo_url = Current folder + /repo
repo_url = "https://github.com/pablotoledo/the-mergementor.git"
Repo.clone_from(repo_url, repo_path)

repo = Repo(repo_path)
assert not repo.bare

# Asegúrate de estar en la rama principal
repo.git.checkout('main')


# Leer contenido de archivos (filtrar por .py como ejemplo)
file_contents = []
for subdir, dirs, files in os.walk(repo_path):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith((".py", ".md", ".txt")):
            with open(filepath, 'r', encoding='utf-8') as f:
                # Guarda una tupla de (ruta del archivo, contenido del archivo)
                file_contents.append((filepath, f.read()))

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Define la función de división en segmentos
def divide_en_segmentos(texto, max_length=512, overlap=50):
    palabras = texto.split()
    segmentos = [' '.join(palabras[i:min(i+max_length, len(palabras))]) for i in range(0, len(palabras), max_length - overlap)]
    return segmentos

# Modifica la función encode_texts para procesar cada segmento de texto
def encode_texts(texts_with_paths, max_length=512, overlap=50):
    results = []
    for filepath, text in texts_with_paths:
        segmentos = divide_en_segmentos(text, max_length, overlap)
        embeddings_segmento = []
        for seg in segmentos:
            encoded_input = tokenizer([seg], padding=True, truncation=True, return_tensors='pt', max_length=max_length)
            with torch.no_grad():
                model_output = model(**encoded_input)
            embeddings_segmento.append(model_output.pooler_output)
        # Promedia los embeddings de los segmentos
        embeddings_promedio = torch.mean(torch.stack(embeddings_segmento), dim=0)
        # Guarda una tupla de (ruta del archivo, embedding promediado)
        results.append((filepath, embeddings_promedio))
    return results

# Repite para el contenido de los archivos si es necesario
file_embeddings = encode_texts(file_contents)

print("")




from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.vectorstores.utils import filter_complex_metadata

class DocumentIndexer:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.embedding_model = FastEmbedEmbeddings()  # Asegúrate de que este modelo sea el adecuado para tus necesidades.

    def ingest_documents(self, files_to_index):
        all_chunks = []
        for file_path in files_to_index:
            loader = TextLoader(file_path=file_path)
            doc = loader.load()  # Aquí asumimos que `load()` devuelve un solo bloque de texto.
            
            # Si doc no es una cadena de texto, deberías ajustar este código.
            chunks = self.text_splitter.split_documents(doc)
            all_chunks.extend(chunks)

        # Ahora, todos los chunks están en all_chunks.
        # A continuación, calculamos embeddings para cada chunk.
        documents_with_embeddings = []
        for chunk in all_chunks:
            embedding = self.embedding_model.get_embedding(chunk)
            documents_with_embeddings.append({"text": chunk, "embedding": embedding})

        # Carga los embeddings en Chroma. Ajusta según la versión y API de Chroma que estés utilizando.
        vector_store = Chroma.from_documents(documents=documents_with_embeddings)

        return vector_store

# Uso de ejemplo:
indexer = DocumentIndexer()
vector_store = indexer.ingest_documents(files_to_index)