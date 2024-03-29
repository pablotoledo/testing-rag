import os
import shutil
from git import Repo
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl

repo_path = os.getcwd() + "/repo"
DATA_PATH = 'repo/'
DB_FAISS_PATH = 'vectorstore/db_faiss'
custom_prompt_template = """Use the following pieces of information to answer the user's question. You are an expert developer who guides about the repository associated to this information. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def download_repo(repo_url="https://github.com/pablotoledo/azure-devops-aks-cluster.git", branch="main"):
    if os.path.exists("repo"):
        shutil.rmtree("repo")
    Repo.clone_from(repo_url, repo_path)
    repo = Repo(repo_path)
    assert not repo.bare
    repo.git.checkout(branch)

def create_vector_db():
    loader = DirectoryLoader(DATA_PATH, silent_errors=True, loader_cls=TextLoader, exclude=['**/.git/**', '**/LICENSE', '**/*.drawio'])
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 2}), return_source_documents=True, chain_type_kwargs={'prompt': prompt})
    return qa_chain

def load_llm():
    llm = CTransformers(model="TheBloke/Llama-2-7B-Chat-GGML", model_type="llama", max_new_tokens=512, temperature=0.5)
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)
    return qa

def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

if __name__ == "__main__":
    create_vector_db()
    print(final_result("describe me the project solution or purpose"))



#chainlit code
@cl.on_chat_start
async def start():
    download_repo()
    create_vector_db()
    chain = qa_bot()
    msg = cl.Message(content="Starting the bot...")
    await msg.send()
    msg.content = "Hi, Welcome to RAG Bot for repositories. What is your query?"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain") 
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    sources = res["source_documents"]

    if sources:
        answer += f"\nSources:" + str(sources)
    else:
        answer += "\nNo sources found"

    await cl.Message(content=answer).send()