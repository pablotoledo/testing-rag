import os
from git import Repo

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


