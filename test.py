from pathlib import Path
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
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        response = requests.post(f"{self.base_url}/chat/completions", json=payload)
        return response.json()["choices"][0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "lmstudio-llm"


def clear_vector_store(vector_store: QdrantVectorStore) -> None:
    """Clear all documents from the vector store."""
    try:
        client = vector_store.client
        client.delete_collection(COLLECTION_NAME)
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
        logger.info("Vector store collection cleared successfully.")
    except Exception as e:
        logger.error(f"Error clearing vector store: {e}")
        raise


def configure_qdrant() -> QdrantVectorStore:
    """Configure and return a Qdrant vector store."""
    logger.info("Configuring Qdrant client and vector store...")

    # Инициализация модели эмбедингов
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", encode_kwargs={"truncation": True, "max_length": 512})
    embeddings.embed_query("test")  # Прогрев модели

    # Создание клиента Qdrant
    client = QdrantClient(path="/tmp/langchain_qdrant")

    # Создание коллекции
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    except Exception:
        logger.info("Collection already exists. It will be used.")

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
            if file.endswith((".txt", ".md", ".py", ".js", ".java", ".cpp", ".h", ".html", ".css")):
                file_path = os.path.join(root, file)
                try:
                    loader = TextLoader(file_path, encoding="utf-8")
                    loaded_docs = loader.load()
                    for doc in loaded_docs:
                        doc.metadata["source_file"] = file_path
                    documents.extend(loaded_docs)
                    logger.info(f"Loaded {file_path}")
                except Exception as e:
                    logger.warning(f"Could not load {file_path}: {e}")

    return documents


def add_repository_to_store(
    vector_store: QdrantVectorStore, 
    repo_url: str, 
    chunk_size: int = 512, 
    chunk_overlap: int = 50,
    clear_existing: bool = True
) -> None:
    """Add repository documents to vector store."""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Очищаем хранилище, если требуется
            if clear_existing:
                clear_vector_store(vector_store)

            # Клонируем репозиторий
            repo_path = clone_github_repo(repo_url, temp_dir)

            # Загружаем документы
            documents = load_documents_from_repo(repo_path)

            if not documents:
                raise ValueError("No documents found in repository")

            # Разделяем документы на чанки
            splitter = SentenceTransformersTokenTextSplitter(
                tokens_per_chunk=chunk_size, chunk_overlap=chunk_overlap, model_name="BAAI/bge-m3"
            )
            texts = splitter.split_documents(documents)

            # Добавляем в векторное хранилище
            ids = [str(uuid4()) for _ in texts]
            vector_store.add_documents(documents=texts, ids=ids)
            logger.info(f"Added {len(texts)} documents from repository to vector store")

        except Exception as e:
            logger.error(f"Error processing repository: {e}")
            raise


def load_or_create_vector_store(
    vector_store: QdrantVectorStore, 
    file_path: str = None, 
    chunk_size: int = 512, 
    chunk_overlap: int = 50,
    clear_existing: bool = True
) -> QdrantVectorStore:
    """Load or create vector store with documents."""
    if file_path:
        logger.info("Loading documents into vector store...")
        
        # Очищаем хранилище, если требуется
        if clear_existing:
            clear_vector_store(vector_store)

        # Загрузка документов
        loader = TextLoader(file_path)
        documents = loader.load()
        for doc in documents:
            doc.metadata["source_file"] = file_path

        # Разделение документов на чанки
        splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=chunk_size, chunk_overlap=chunk_overlap, model_name="BAAI/bge-m3"
        )
        texts = splitter.split_documents(documents)

        # Добавление документов в хранилище
        ids = [str(uuid4()) for _ in texts]
        vector_store.add_documents(documents=texts, ids=ids)
        logger.info(f"Added {len(texts)} documents to vector store.")
    else:
        client = vector_store.client
        collection_info = client.get_collection(COLLECTION_NAME)
        logger.info(f"Using existing vector store with {collection_info.vectors_count} vectors.")

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


# === Функции для Gradio ===
def answer_question(question, temperature, max_tokens, top_p, k_results, fetch_k, mmr_lambda):
    # Обновляем параметры LLM
    llm.temperature = temperature
    llm.max_tokens = max_tokens
    llm.top_p = top_p

    # Настраиваем ретривер с новыми параметрами
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": k_results, "fetch_k": fetch_k, "lambda_mult": mmr_lambda}
    )

    # Сборка цепочки с текущими параметрами
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_template},
        return_source_documents=True,
    )

    response = qa_chain({"query": question})

    # Получаем ответ и исходные документы
    answer = response["result"]
    source_docs = response["source_documents"]

    # Формируем список уникальных исходных файлов
    source_files = set()
    for doc in source_docs:
        if "source_file" in doc.metadata:
            source_files.add(doc.metadata["source_file"])

    # Формируем итоговый ответ с указанием источников
    if source_files:
        answer += "\n\nИсточники:\n- " + "\n- ".join(source_files)

    return answer


def add_repository(repo_url, chunk_size, chunk_overlap, clear_existing):
    try:
        add_repository_to_store(vector_store, repo_url, chunk_size, chunk_overlap, clear_existing)
        return "Репозиторий успешно добавлен!" + (" (старые данные удалены)" if clear_existing else "")
    except Exception as e:
        return f"Ошибка при добавлении репозитория: {str(e)}"


# === Кастомный промпт ===
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Твоя задача - анализировать загруженный проект и отвечать на вопросы пользователя касательно проекта\n\n"
        "Контекст:\n{context}\n\nВопрос: {question}\nОтвет:"
    ),
)

# === Интерфейс Gradio ===
css_file = Path("./gradio_style.css")

with gr.Blocks(css=css_file.read_text() if css_file.exists() else "") as demo:
    with gr.Tabs():
        with gr.Tab("Добавить репозиторий", id="repo_tab"):
            with gr.Row():
                with gr.Column():
                    repo_url_input = gr.Textbox(
                        placeholder="https://github.com/username/repository.git", label="URL GitHub репозитория"
                    )
                    with gr.Accordion("Настройки чанкинга", open=False):
                        chunk_size = gr.Slider(128, 1024, value=512, step=64, label="Размер чанка (токены)")
                        chunk_overlap = gr.Slider(0, 256, value=50, step=16, label="Перекрытие чанков (токены)")
                    clear_existing = gr.Checkbox(value=True, label="Очистить существующие данные")
                    add_repo_button = gr.Button("Добавить репозиторий", variant="primary")
                with gr.Column():
                    repo_status_output = gr.Textbox(label="Статус", interactive=False)

        with gr.Tab("Задать вопрос", id="question_tab"):
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(lines=2, placeholder="Задай вопрос по проекту...", label="Вопрос")
                    with gr.Accordion("Настройки модели", open=False):
                        temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
                        max_tokens = gr.Slider(128, 4096, value=1024, step=128, label="Max Tokens")
                        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.1, label="Top-p")
                    with gr.Accordion("Настройки поиска", open=False):
                        k_results = gr.Slider(1, 10, value=5, step=1, label="Количество результатов")
                        fetch_k = gr.Slider(5, 50, value=20, step=5, label="Fetch K (кол-во кандидатов)")
                        mmr_lambda = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="MMR Lambda (разнообразие)")
                    ask_button = gr.Button("Спросить", variant="primary")
                with gr.Column():
                    answer_output = gr.Textbox(label="Ответ", interactive=False)

        with gr.Tab("Информация", id="info_tab"):
            gr.Markdown("### Инструкция по использованию")
            gr.Markdown(
                """
            1. **Добавить репозиторий** - загрузите новый репозиторий для анализа
            2. **Задать вопрос** - задавайте вопросы о вашем проекте
            3. Настройте параметры модели и поиска для лучших результатов
            """
            )
            gr.Markdown("### Технологии")
            gr.Markdown(
                """
            - **LM Studio** - локальный LLM
            - **LangChain** - обработка и анализ текста
            - **Qdrant** - векторное хранилище
            - **bge-m3** - модели эмбедингов
            """
            )

    ask_button.click(
        answer_question,
        inputs=[question_input, temperature, max_tokens, top_p, k_results, fetch_k, mmr_lambda],
        outputs=answer_output,
    )
    add_repo_button.click(
        add_repository, 
        inputs=[repo_url_input, chunk_size, chunk_overlap, clear_existing], 
        outputs=repo_status_output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=True
    )