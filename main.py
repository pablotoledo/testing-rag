from git import Repo

# Clona el repositorio si es necesario
repo_path = "path/to/your/repository"
repo_url = "url_of_the_repository"
Repo.clone_from(repo_url, repo_path)

repo = Repo(repo_path)
assert not repo.bare

# Asegúrate de estar en la rama principal
repo.git.checkout('main')

# Ejemplo de cómo leer archivos o mensajes de commit
commits = list(repo.iter_commits('main', max_count=100))  # Limita a los últimos 100 por simplicidad
commit_messages = [commit.message for commit in commits]

# Leer contenido de archivos (filtrar por .py como ejemplo)
import os

file_contents = []
for subdir, dirs, files in os.walk(repo_path):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".py"):
            with open(filepath, 'r', encoding='utf-8') as f:
                file_contents.append(f.read())

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def encode_texts(texts):
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    return embeddings

# Ejemplo de codificación de mensajes de commit
commit_embeddings = encode_texts(commit_messages)
# Repite para el contenido de los archivos si es necesario
file_embeddings = encode_texts(file_contents)

