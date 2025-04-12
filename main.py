import os
import requests
import gradio as gr
import logging
from typing import Optional, List
from uuid import uuid4
import subprocess
import tempfile

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.language_models.llms import LLM
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "code-assistant-store")

class LMStudioLLM(LLM):
    model: str = "local-model"
    base_url: str = "http://127.0.0.1:1234/v1"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "lmstudio-llm"

def configure_qdrant() -> QdrantVectorStore:
    """Configure and return a Qdrant vector store."""
    logger.info("Configuring Qdrant client and vector store...")
    
    # Инициализация модели эмбедингов
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        encode_kwargs={'truncation': True, 'max_length': 512}
    )
    embeddings.embed_query("test")  # Прогрев модели

    # Создание клиента Qdrant
    client = QdrantClient(path='/tmp/langchain_qdrant')

    # Создание коллекции, если она не существует
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    except ValueError:
        logger.info("Collection already exists. Skipping creation.")

    # Создание векторного хранилища
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )
    
    logger.info("Qdrant configuration completed.")
    return vector_store

def clone_github_repo(repo_url: str, temp_dir: str) -> str:
    """Clone GitHub repository to temporary directory."""
    try:
        repo_name = repo_url.split("/")[-1]
        if repo_name.endswith(".git"):
            repo_name = repo_name[:-4]
        repo_path = os.path.join(temp_dir, repo_name)
        
        subprocess.run(["git", "clone", repo_url, repo_path], check=True)
        logger.info(f"Repository cloned successfully to {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e}")
        raise

def load_documents_from_repo(repo_path: str) -> List[Document]:
    """Load documents from repository."""
    documents = []
    
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.txt', '.md', '.py', '.js', '.java', '.cpp', '.h', '.html', '.css')):
                file_path = os.path.join(root, file)
                try:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents.extend(loader.load())
                    logger.info(f"Loaded {file_path}")
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")
    
    return documents

def add_repository_to_store(vector_store: QdrantVectorStore, repo_url: str) -> None:
    """Add repository documents to vector store."""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Клонируем репозиторий
            repo_path = clone_github_repo(repo_url, temp_dir)
            
            # Загружаем документы
            documents = load_documents_from_repo(repo_path)
            
            if not documents:
                raise ValueError("No documents found in repository")
            
            # Разделяем документы на чанки
            splitter = SentenceTransformersTokenTextSplitter(
                tokens_per_chunk=512,
                chunk_overlap=50,
                model_name="BAAI/bge-m3"
            )
            texts = splitter.split_documents(documents)
            
            # Добавляем в векторное хранилище
            ids = [str(uuid4()) for _ in texts]
            vector_store.add_documents(documents=texts, ids=ids)
            logger.info(f"Added {len(texts)} documents from repository to vector store")
            
        except Exception as e:
            logger.error(f"Error processing repository: {e}")
            raise

def load_or_create_vector_store(vector_store: QdrantVectorStore, file_path: str = None) -> QdrantVectorStore:
    """Load or create vector store with documents."""
    # Проверяем, есть ли уже документы в хранилище
    client = vector_store.client
    collection_info = client.get_collection(COLLECTION_NAME)
    
    if collection_info.vectors_count == 0 and file_path:
        logger.info("No documents in vector store. Creating new index...")
        
        # Загрузка документов
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Разделение документов на чанки
        splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=512,
            chunk_overlap=50,
            model_name="BAAI/bge-m3"
        )
        texts = splitter.split_documents(documents)
        
        # Добавление документов в хранилище
        ids = [str(uuid4()) for _ in texts]
        vector_store.add_documents(documents=texts, ids=ids)
        logger.info(f"Added {len(texts)} documents to vector store.")
    else:
        logger.info(f"Vector store already contains {collection_info.vectors_count} vectors. Using existing index.")
    
    return vector_store

# === Инициализация компонентов ===
llm = LMStudioLLM()
vector_store = configure_qdrant()

# Загрузка или создание векторного хранилища
FILE_PATH = r"./my_notes.txt"
if os.path.exists(FILE_PATH):
    vector_store = load_or_create_vector_store(vector_store, FILE_PATH)
else:
    vector_store = load_or_create_vector_store(vector_store)

# Настройка ретривера
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)

# === Кастомный промпт ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Твоя задача - анализировать загруденный проект и отвечать на вопросы пользователя касательно проекта\n\n"
        "Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    ),
)

# === Сборка цепочки ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=False,
)

# === Функции для Gradio ===
def answer_question(question):
    response = qa_chain({"query": question})
    return response["result"]

def add_repository(repo_url):
    try:
        add_repository_to_store(vector_store, repo_url)
        return "Репозиторий успешно добавлен!"
    except Exception as e:
        return f"Ошибка при добавлении репозитория: {str(e)}"

# === Интерфейс Gradio ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Локальный ассистент по коду")
    gr.Markdown("Использует LLaMA 3.1 через LM Studio + LangChain + Qdrant + bge-m3")
    
    with gr.Tab("Задать вопрос"):
        question_input = gr.Textbox(lines=2, placeholder="Задай вопрос по проекту...", label="Вопрос")
        answer_output = gr.Textbox(label="Ответ")
        ask_button = gr.Button("Спросить")
    
    with gr.Tab("Добавить репозиторий"):
        repo_url_input = gr.Textbox(
            placeholder="https://github.com/username/repository.git",
            label="URL GitHub репозитория"
        )
        add_repo_button = gr.Button("Добавить репозиторий")
        repo_status_output = gr.Textbox(label="Статус")
    
    ask_button.click(answer_question, inputs=question_input, outputs=answer_output)
    add_repo_button.click(add_repository, inputs=repo_url_input, outputs=repo_status_output)

# === Запуск веб-интерфейса ===
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)