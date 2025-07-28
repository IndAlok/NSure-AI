# rag_core.py
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils import get_pdf_text_from_url

class RAGCore:
    def __init__(self, document_url: str, embedding_model, llm):
        print(f"--- Initializing RAG Core for document: {document_url} ---")
        self.llm = llm

        raw_text = get_pdf_text_from_url(document_url)
        if not raw_text:
            raise ValueError("Failed to retrieve or parse document text.")

        print("Step 1: Chunking document text...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = text_splitter.split_text(raw_text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(f"   -> Created {len(documents)} document chunks.")

        print("Step 2: Embedding chunks and creating FAISS index...")
        self.vector_store = FAISS.from_documents(documents, embedding_model)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) # Reduced k for speed
        print("   -> FAISS index created successfully.")

        prompt_template = """
        You are an expert AI assistant for insurance policies. Answer the user's question based ONLY on the provided context.
        - Be concise and direct.
        - If the information is not in the context, respond with: "Information not found in the provided policy document."

        CONTEXT:
        {context}

        QUESTION:
        {input}

        ANSWER:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.qa_chain = create_stuff_documents_chain(self.llm, prompt)
        print("--- RAG Core for this document is ready. ---")

    def answer_question(self, question: str) -> str:
        print(f"Received query: '{question}'")
        relevant_docs = self.retriever.invoke(question)
        response = self.qa_chain.invoke({
            "input": question,
            "context": relevant_docs
        })
        return response.strip()
