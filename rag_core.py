# rag_core.py
import re
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils import get_pdf_text_from_url

def intelligent_chunking(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Performs a two-stage intelligent chunking process.
    1. Splits the document into logical sections based on common policy structures.
    2. Further splits any oversized sections recursively.
    """
    # Stage 1: Split by logical sections (clauses, sections, numbered points)
    # This regex looks for patterns like "1. ", "a. ", "Section X", "Clause Y", etc.
    # or multiple newlines which often separate paragraphs.
    logical_splits = re.split(r'(?m)(^\s*\d+\.\s|^\s*[a-zA-Z]\.\s|^\s*•\s|^\s*Section\s\w+|^\s*Clause\s\w+|\n\s*\n)', text)
    
    # Combine the delimiter with the text that follows it
    combined_splits = []
    i = 1
    while i < len(logical_splits):
        # Combine the delimiter (e.g., "1. ") with the subsequent text
        combined_text = (logical_splits[i] + logical_splits[i+1]).strip()
        if combined_text:
            combined_splits.append(combined_text)
        i += 2
    
    # Add the initial part of the text if it exists
    if logical_splits and logical_splits[0].strip():
        combined_splits.insert(0, logical_splits[0].strip())

    # Stage 2: Use RecursiveCharacterTextSplitter on any chunks that are too large
    final_chunks = []
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    for chunk in combined_splits:
        if len(chunk) > chunk_size:
            # If a logical chunk is too big, split it further
            sub_chunks = recursive_splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
            
    return final_chunks

class RAGCore:
    def __init__(self, document_url: str, embedding_model, llm):
        print(f"--- Initializing RAG Core for document: {document_url} ---")
        self.llm = llm

        raw_text = get_pdf_text_from_url(document_url)
        if not raw_text:
            raise ValueError("Failed to retrieve or parse document text.")

        # Step 1: Perform Intelligent Chunking
        print("Step 1: Performing intelligent, two-stage chunking...")
        chunks = intelligent_chunking(raw_text, chunk_size=1000, chunk_overlap=200)
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(f"   -> Created {len(documents)} high-quality document chunks.")

        # Step 2: Embed chunks and create FAISS index
        print("Step 2: Embedding chunks and creating FAISS index...")
        self.vector_store = FAISS.from_documents(documents, embedding_model)
        # Give the LLM more context to ensure it finds the answer.
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("   -> FAISS index created successfully.")

        # A more robust prompt to guide the LLM
        prompt_template = """
        You are a highly precise AI assistant for analyzing insurance policies.
        Your task is to provide a clear and accurate answer to the user's question based *only* on the provided context.
        - The answer may be synthesized from multiple parts of the context.
        - If the information to answer the question is not present in the context, you MUST respond with: "Information not found in the provided policy document."
        - Be direct and do not add any conversational fluff.

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
        
        # For debugging, you can print the retrieved context
        # print("--- Retrieved Context ---")
        # for doc in relevant_docs:
        #     print(doc.page_content)
        #     print("---")

        response = self.qa_chain.invoke({
            "input": question,
            "context": relevant_docs
        })
        return response.strip()
