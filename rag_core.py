import re
import asyncio
import numpy as np
import pickle
from typing import List, Tuple
from functools import lru_cache
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from utils import get_pdf_text_from_url
from rank_bm25 import BM25Okapi
from database import db_cache

@lru_cache(maxsize=64)
def context_aware_chunk_text(text_hash: str, text: str, chunk_size: int = 700, chunk_overlap: int = 80) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ";", ":", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    cleaned_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    cleaned_text = ' '.join(cleaned_sentences)
    
    chunks = splitter.split_text(cleaned_text)
    return [chunk.strip() for chunk in chunks if len(chunk.strip()) > 20]

class HybridRetriever:
    def __init__(self, docs: List[Document], embedding_model):
        self.docs = docs
        self.embedding_model = embedding_model
        self.bm25 = None
        self.doc_embeddings = None
        self._setup_retrievers()
    
    def _setup_retrievers(self):
        if not self.docs:
            return
            
        texts = [doc.page_content for doc in self.docs]
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        try:
            self.doc_embeddings = self.embedding_model.embed_documents(texts)
        except Exception:
            self.doc_embeddings = None
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        return np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-8)
    
    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        if not self.docs:
            return []
        
        results = []
        scores = {}
        
        if self.bm25:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            for i, score in enumerate(bm25_scores):
                if score > 0:
                    scores[i] = scores.get(i, 0) + score * 0.4
        
        if self.doc_embeddings:
            try:
                query_embedding = self.embedding_model.embed_query(query)
                for i, doc_emb in enumerate(self.doc_embeddings):
                    sim = self._cosine_similarity(query_embedding, doc_emb)
                    scores[i] = scores.get(i, 0) + sim * 0.6
            except Exception:
                pass
        
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        seen = set()
        for i in sorted_indices[:k]:
            content_hash = hash(self.docs[i].page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                results.append(self.docs[i])
        
        return results

class OptimizedRAGCore:
    def __init__(self, embedding_model, llm):
        self.embedding_model = embedding_model
        self.llm = llm
        
        self.prompt_template = ChatPromptTemplate.from_template("""
Extract the key information to answer the question. Write one clear sentence with specific details.

Examples:
Question: "What is the grace period for premium payment?"
Answer: "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits."

Question: "What is the waiting period for pre-existing diseases?"
Answer: "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered."

Context: {context}

Question: {question}

Answer (one clear sentence with specific details):""")

    async def _get_answer_for_question(self, question: str, retriever: HybridRetriever) -> Tuple[str, str]:
        relevant_docs = retriever.retrieve(question, k=8)
        context = "\n\n".join([doc.page_content for doc in relevant_docs[:4]])
        
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, 
            self._get_answer_from_llm, 
            context, 
            question
        )
        
        clean_answer = self._format_answer(response)
        return question, clean_answer

    async def process_queries(self, pdf_url: str, questions: List[str]) -> List[str]:
        try:
            cached_answers = {}
            uncached_questions = []
            
            cache_results = await asyncio.gather(*[db_cache.get_query_cache(pdf_url, q) for q in questions])

            for question, cached_answer in zip(questions, cache_results):
                if cached_answer:
                    cached_answers[question] = cached_answer
                else:
                    uncached_questions.append(question)
            
            if not uncached_questions:
                return [cached_answers[q] for q in questions]
            
            doc_cache_data = await db_cache.get_doc_cache(pdf_url)
            
            if doc_cache_data:
                text = doc_cache_data['text']
                chunks = doc_cache_data['chunks']
            else:
                text = get_pdf_text_from_url(pdf_url)
                if not text:
                    return ["Error: Could not extract text from PDF"] * len(questions)
                
                text_hash = hash(text[:1000])
                chunks = context_aware_chunk_text(str(text_hash), text)
                
                embeddings_bytes = pickle.dumps([])
                await db_cache.set_doc_cache(pdf_url, text, chunks, embeddings_bytes)
            
            docs = [Document(page_content=chunk) for chunk in chunks]
            retriever = HybridRetriever(docs, self.embedding_model)
            
            tasks = [self._get_answer_for_question(q, retriever) for q in uncached_questions]
            new_results = await asyncio.gather(*tasks)

            cache_tasks = []
            for question, answer in new_results:
                cached_answers[question] = answer
                cache_tasks.append(db_cache.set_query_cache(pdf_url, question, answer))
            await asyncio.gather(*cache_tasks)
            
            return [cached_answers[q] for q in questions]
            
        except Exception as e:
            return [f"Error: {str(e)}"] * len(questions)

    def _get_answer_from_llm(self, context: str, question: str) -> str:
        try:
            prompt = self.prompt_template.format(context=context[:2000], question=question)
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def _format_answer(self, raw_answer: str) -> str:
        answer = raw_answer.strip()
        answer = answer.replace('\\"', '"')
        answer = re.sub(r'\s+', ' ', answer)
        answer = re.sub(r'^(Answer|Response|Based on|According to)[:\s]*', '', answer, flags=re.IGNORECASE).strip()
        
        if not answer.endswith(('.', '!', '?')):
            answer += '.'
        
        return answer[:400] if len(answer) > 400 else answer