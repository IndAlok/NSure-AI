import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from utils import get_pdf_text_from_url

def smart_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    # Enhanced regex for better section detection including medical and policy conditions
    logical_splits = re.split(r'(?m)(^\s*\d+\.\s|^\s*[a-zA-Z]\.\s|^\s*•\s|^\s*Section\s\w+|^\s*Clause\s\w+|^\s*Article\s\w+|^\s*\d+\.\d+\s|^\s*Sub-Limit|^\s*Waiting Period|^\s*Specific|^\s*Treatment|^\s*Surgery|^\s*Condition|\n\s*\n)', text)
    
    combined_splits = []
    i = 1
    while i < len(logical_splits):
        combined_text = (logical_splits[i] + logical_splits[i+1]).strip()
        if combined_text:
            combined_splits.append(combined_text)
        i += 2
    
    if logical_splits and logical_splits[0].strip():
        combined_splits.insert(0, logical_splits[0].strip())

    final_chunks = []
    # Optimize chunks for medical conditions and policy terms
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n", 
            "\nWaiting Period", "\nSub-Limit", "\nCoverage", "\nTreatment", "\nCondition",
            "\n", ". ", "; ", ", ", " ", ""
        ]  # Policy and medical condition-aware separators
    )
    
    for chunk in combined_splits:
        if len(chunk) > chunk_size:
            sub_chunks = recursive_splitter.split_text(chunk)
            final_chunks.extend(sub_chunks)
        else:
            final_chunks.append(chunk)
    
    print(f"Split into {len(final_chunks)} policy-aware chunks")
    return final_chunks

class RAGCore:
    def __init__(self, document_url: str, embedding_model, llm):
        print(f"🔄 Setting up RAG for: {document_url}")
        self.llm = llm

        raw_text = get_pdf_text_from_url(document_url)
        if not raw_text:
            raise ValueError("Couldn't read PDF - check the URL")

        print("📝 Chunking document...")
        chunks = smart_chunk_text(raw_text, chunk_size=800, chunk_overlap=150)
        documents = [Document(page_content=chunk) for chunk in chunks]
        print(f"✅ Created {len(documents)} chunks")

        print("🧠 Building vector index...")
        self.vector_store = FAISS.from_documents(documents, embedding_model)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        print("✅ Vector index ready")

        prompt_template = """You are a knowledgeable insurance policy assistant specialized in answering insurance policy queries.
Answer questions clearly, accurately, and concisely based only on the provided policy context.
Provide factual, professional responses suitable for policyholders or agents.

FORMATTING RULES:
- Keep answers concise but complete with essential details
- Include specific numbers, timeframes, and conditions exactly as stated
- For yes/no questions, start with "Yes" or "No" then provide key details
- Limit to 1-2 sentences for most answers
- Focus on the most important facts that directly answer the question
- Understand technical terms related to the question and answer accordingly
- Avoid unnecessary explanatory phrases

Examples of desired format:
Q: What is the grace period for premium payment?
A: A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.

Q: What is the waiting period for pre-existing diseases?
A: There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.

Q: Does this policy cover maternity expenses?
A: Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months.

Context from insurance policy: {context}

Question: {input}

Answer:"""

        prompt = ChatPromptTemplate.from_template(prompt_template)
        self.qa_chain = create_stuff_documents_chain(self.llm, prompt)
        print("✅ QA system ready")

    def answer_question(self, question: str) -> str:
        print(f"❓ Question: {question}")
        
        docs = self.retriever.invoke(question)
        
        response = self.qa_chain.invoke({
            "input": question,
            "context": docs
        })
        
        return response.strip()

    def batch_retrieve_documents(self, questions: List[str]) -> Dict[str, List[Document]]:
        """Batch retrieve documents for multiple questions efficiently"""
        print(f"🔍 Batch retrieving documents for {len(questions)} questions...")
        
        # Use ThreadPoolExecutor for parallel retrieval
        with ThreadPoolExecutor(max_workers=min(len(questions), 5)) as executor:
            results = list(executor.map(
                lambda q: (q, self.retriever.invoke(q)), 
                questions
            ))
        
        return dict(results)

    def batch_answer_questions(self, questions: List[str]) -> List[str]:
        """Process multiple questions in batch with Gemini 1.5 Flash"""
        if not questions:
            return []
        
        # Handle single question case
        if len(questions) == 1:
            return [self.answer_question(questions[0])]
        
        print(f"🚀 Processing {len(questions)} questions with Gemini 1.5 Flash...")
        
        # Step 1: Batch retrieve all documents
        question_docs = self.batch_retrieve_documents(questions)
        
        # Step 2: Prepare all QA inputs
        qa_inputs = []
        for question in questions:
            qa_inputs.append({
                "input": question,
                "context": question_docs[question]
            })
        
        # Step 3: Process with controlled concurrency for Gemini 1.5 Flash
        # Gemini 1.5 Flash has good performance with moderate rate limits
        max_workers = min(len(questions), 4)  # Conservative concurrency for 1.5 Flash
        print(f"🧠 Processing answers with {max_workers} workers (Gemini 1.5 Flash)...")
        
        answers = [None] * len(qa_inputs)
        
        def process_with_retry(index_input_pair):
            index, qa_input = index_input_pair
            max_retries = 3  # More retries for 1.5 Flash
            base_delay = 0.3  # Slightly longer delay for 1.5 Flash
            
            for attempt in range(max_retries):
                try:
                    # Add delay between requests to avoid rate limits
                    if attempt > 0:
                        delay = base_delay * (2 ** attempt)
                        print(f"⏳ Retry {attempt + 1} for question {index + 1}, waiting {delay}s...")
                        time.sleep(delay)
                    
                    response = self.qa_chain.invoke(qa_input)
                    return index, response.strip()
                    
                except Exception as e:
                    if "rate" in str(e).lower() or "429" in str(e) or "quota" in str(e).lower():
                        print(f"⚠️  Rate limit hit for question {index + 1}, attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** (attempt + 1))
                            time.sleep(delay)
                            continue
                    elif "api" in str(e).lower() and ("key" in str(e).lower() or "auth" in str(e).lower()):
                        print(f"❌ API key error for question {index + 1}: {e}")
                        return index, "Error: Please check your Gemini API key configuration."
                    
                    print(f"❌ Question {index + 1} failed after {attempt + 1} attempts: {e}")
                    return index, f"Error: {str(e)}"
            
            return index, "Error: Max retries exceeded"
        
        # Process with controlled threading - Gemini 1.5 Flash moderate concurrency
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Moderate delays for Gemini 1.5 Flash
            futures = []
            for i, qa_input in enumerate(qa_inputs):
                future = executor.submit(process_with_retry, (i, qa_input))
                futures.append(future)
                
                # Moderate delay for Gemini 1.5 Flash
                if i < len(qa_inputs) - 1:
                    time.sleep(0.1)  # 100ms delay between submissions
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    index, answer = future.result()
                    answers[index] = answer
                except Exception as e:
                    print(f"❌ Future failed: {e}")
        
        # Fill any None values with error messages
        for i in range(len(answers)):
            if answers[i] is None:
                answers[i] = "Error: Processing failed"
        
        print(f"✅ Completed batch processing of {len(answers)} questions with Gemini 1.5 Flash")
        return answers