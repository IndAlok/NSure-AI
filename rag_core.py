# rag_core.py
"""
This module defines the core Retrieval-Augmented Generation (RAG) pipeline.
It encapsulates the logic for document chunking, embedding, retrieval, and
the final answer generation using an LLM.
"""
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

from utils import get_pdf_text_from_url

# Load environment variables from the .env file (for OPENAI_API_KEY)
load_dotenv()

class RAGCore:
    """
    A class that encapsulates the entire RAG pipeline from document loading
    to question answering.
    """
    def __init__(self, document_url: str):
        """
        Initializes the RAG pipeline by processing the document at the given URL.
        
        Args:
            document_url (str): The URL of the document to process.
        """
        print("--- Initializing RAG Core Pipeline ---")
        
        # 1. Load and Process Document
        raw_text = get_pdf_text_from_url(document_url)
        if not raw_text:
            raise ValueError("Failed to retrieve or parse document text.")
        
        # 2. Chunk the document for effective retrieval
        # This splitter attempts to keep paragraphs/sentences together, which is
        # ideal for semantic context.
        print("Step 1: Chunking document text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150, # A small overlap helps maintain context between chunks
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", ". ", " ", ""] # Splits by paragraphs first, then lines, etc.
        )
        chunks = text_splitter.split_text(raw_text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(f"   -> Created {len(documents)} document chunks.")

        # 3. Embed chunks using a local, fast model and create FAISS vector store
        print("Step 2: Embedding chunks and creating FAISS index...")
        # Using a local model is faster and free. 'all-MiniLM-L6-v2' is a great balance
        # of speed and performance.
        embedding_model = HuggingFaceEmbeddings(
            model_name="./models/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'} # Explicitly use CPU
        )
        
        # This creates the vector store in memory for ultra-fast retrieval
        self.vector_store = FAISS.from_documents(documents, embedding_model)
        
        # The retriever will fetch the top 'k' most relevant chunks for a given query
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("   -> FAISS index created successfully in memory.")

        # 4. Define the LLM and the prompt template for answering questions
        print("Step 3: Configuring LLM and prompt template...")
        # Using gpt-4-turbo for a good balance of intelligence and speed.
        # Low temperature makes the output more deterministic and factual.
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        
        # This prompt is crucial for accuracy, efficiency, and explainability.
        # It forces the LLM to use ONLY the provided context.
        prompt_template = """
        You are a highly intelligent AI assistant specializing in analyzing insurance policy documents.
        Your task is to answer the user's question based *only* on the provided context below.
        - Be concise, clear, and direct.
        - If the information to answer the question is not found in the context, you MUST respond with: "Information not found in the provided policy document."
        - Do not use any external knowledge or make assumptions.

        CONTEXT:
        {context}

        QUESTION:
        {input}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # 5. Create the final question-answering chain
        self.qa_chain = create_stuff_documents_chain(self.llm, prompt)
        print("--- RAG Core Pipeline is ready for queries. ---")

    def answer_question(self, question: str) -> str:
        """
        Takes a user question, retrieves relevant context from the document,
        and generates a precise answer.

        Args:
            question (str): The user's question.

        Returns:
            str: The generated answer from the LLM.
        """
        print(f"Received query: '{question}'")
        # 1. Retrieve relevant documents from the FAISS vector store
        relevant_docs = self.retriever.invoke(question)
        
        # 2. Invoke the chain with the retrieved documents and the question
        # The 'stuff' chain combines all retrieved documents into the context.
        response = self.qa_chain.invoke({
            "input": question,
            "context": relevant_docs
        })
        
        return response.strip()

