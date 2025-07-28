# rag_core.py
"""
This module defines the core Retrieval-Augmented Generation (RAG) pipeline.
It is optimized to accept pre-loaded models for fast instantiation.
"""
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain

from utils import get_pdf_text_from_url

class RAGCore:
    """
    A class that encapsulates the RAG pipeline for a specific document,
    using pre-loaded embedding and LLM models.
    """
    def __init__(self, document_url: str, embedding_model, llm):
        """
        Initializes the RAG pipeline for a specific document.

        Args:
            document_url (str): The URL of the document to process.
            embedding_model: The pre-loaded sentence-transformer embedding model.
            llm: The pre-loaded ChatOpenAI model.
        """
        print(f"--- Initializing RAG Core for document: {document_url} ---")
        self.llm = llm  # Use the passed-in LLM

        # 1. Load and Process Document
        raw_text = get_pdf_text_from_url(document_url)
        if not raw_text:
            raise ValueError("Failed to retrieve or parse document text.")

        # 2. Chunk the document
        print("Step 1: Chunking document text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(raw_text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(f"   -> Created {len(documents)} document chunks.")

        # 3. Embed chunks and create FAISS vector store using the pre-loaded model
        print("Step 2: Embedding chunks and creating FAISS index...")
        self.vector_store = FAISS.from_documents(documents, embedding_model)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("   -> FAISS index created successfully in memory.")

        # 4. Define the prompt template

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
        # 5. Create the final question-answering chain

        from langchain_core.prompts import ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.qa_chain = create_stuff_documents_chain(self.llm, prompt)
        print("--- RAG Core for this document is ready. ---")

    def answer_question(self, question: str) -> str:
        """
        Takes a user question, retrieves relevant context, and generates an answer.
        """
        print(f"Received query: '{question}'")
        relevant_docs = self.retriever.invoke(question)
        response = self.qa_chain.invoke({
            "input": question,
            "context": relevant_docs
        })
        return response.strip()
