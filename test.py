import os
import logging
import tempfile
from typing import List

import gradio as gr
from git import Repo
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from qdrant_client import QdrantClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Configuration
EMBEDDING_MODEL = "BAAI/bge-m3"
COLLECTION_NAME = "github-repo-qa"
LOCAL_LLM_ENDPOINT = "http://localhost:1234/v1"  # LM Studio default endpoint
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class GitHubRepoQA:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            encode_kwargs={'truncation': True, 'max_length': 512}
        )
        self.client = QdrantClient(path=":memory:")
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        
    def clone_repo(self, repo_url: str, temp_dir: str) -> str:
        """Clone a GitHub repository to a temporary directory."""
        try:
            repo_dir = os.path.join(temp_dir, "repo")
            Repo.clone_from(repo_url, repo_dir)
            logger.info(f"Repository cloned successfully to {repo_dir}")
            return repo_dir
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            raise

    def load_documents(self, repo_path: str) -> List[Document]:
        """Load documents from the repository with language-specific parsing."""
        loader = GenericLoader.from_filesystem(
            repo_path,
            glob="**/*",
            suffixes=[".py", ".js", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp"],
            parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
        )
        return loader.load()

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks with language-aware splitting."""
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        return splitter.split_documents(docs)

    def setup_vector_store(self, docs: List[Document]) -> None:
        """Set up Qdrant vector store with documents."""
        # Create collection if it doesn't exist
        try:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={"size": 1024, "distance": "Cosine"},
            )
        except Exception:
            logger.info("Collection already exists")

        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding=self.embeddings,
        )
        
        # Add documents to the vector store
        self.vector_store.add_documents(docs)
        
        # Configure retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5}
        )

    def setup_qa_chain(self) -> None:
        """Set up the QA chain with the local LLM."""
        template = """Answer the question based only on the following context, 
        which includes source file information. Always cite the source files 
        you used to derive your answer:

        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        self.qa_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self._get_local_llm()
            | StrOutputParser()
        )

    def _get_local_llm(self):
        """Get the local LLM from LM Studio."""
        
        return ChatOpenAI(
            base_url=LOCAL_LLM_ENDPOINT,
            api_key="not-needed",  # LM Studio doesn't require an API key
            model="local-model",
            temperature=0.1,
        )

    def process_repository(self, repo_url: str) -> str:
        """Process a GitHub repository and set up the QA system."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone repository
                repo_path = self.clone_repo(repo_url, temp_dir)
                
                # Load and split documents
                docs = self.load_documents(repo_path)
                split_docs = self.split_documents(docs)
                
                # Setup vector store and QA chain
                self.setup_vector_store(split_docs)
                self.setup_qa_chain()
                
                return "Repository processed successfully! You can now ask questions."
            except Exception as e:
                logger.error(f"Error processing repository: {e}")
                return f"Error: {str(e)}"

    def ask_question(self, question: str) -> str:
        """Ask a question about the repository."""
        if not self.qa_chain:
            return "Please process a repository first."
        
        try:
            return self.qa_chain.invoke(question)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error answering question: {str(e)}"

# Create the QA system instance
qa_system = GitHubRepoQA()

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# GitHub Repository QA System")
    gr.Markdown("Ask questions about a public GitHub repository")
    
    with gr.Row():
        with gr.Column():
            repo_url = gr.Textbox(
                label="GitHub Repository URL",
                placeholder="https://github.com/username/repo",
            )
            process_btn = gr.Button("Process Repository")
            status = gr.Textbox(label="Status")
        
        with gr.Column():
            question = gr.Textbox(
                label="Question",
                placeholder="What does this repository do?",
            )
            ask_btn = gr.Button("Ask Question")
            answer = gr.Textbox(label="Answer", lines=10)
    
    process_btn.click(
        qa_system.process_repository,
        inputs=repo_url,
        outputs=status,
    )
    
    ask_btn.click(
        qa_system.ask_question,
        inputs=question,
        outputs=answer,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)