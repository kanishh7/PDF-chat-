# Enhanced LangChain RAG PDF Reader with Optimizations and Benchmarking (Ollama Version, NumPy-based Dummy Embeddings)

import time
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings.base import Embeddings

class DummyNumpyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [np.random.rand(384).tolist() for _ in texts]

    def embed_query(self, text):
        return np.random.rand(384).tolist()

pdf_path = "example.pdf"
loader = PyPDFLoader(pdf_path)
print("Loading PDF...")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
print("Splitting document into chunks...")
chunks = splitter.split_documents(documents)

print("Generating embeddings...")
embedding_model = DummyNumpyEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding_model)

print("Loading local LLM with Ollama...")
llm = OllamaLLM(model="mistral")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are an expert assistant. Answer the question using only the context below. Do not make up information.
If unsure, say 'I couldn't find the answer in the document.'

Context:
{context}

Question:
{question}
"""
)

retriever = vectorstore.as_retriever(search_type="similarity", k=5)
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template}
)

def inspect_retrieved_docs(query):
    docs = retriever.get_relevant_documents(query)
    print("\nTop retrieved documents:")
    for i, doc in enumerate(docs):
        print(f"\n[Doc {i+1}]\n{doc.page_content[:500]}...")

while True:
    query = input("\nAsk a question about the PDF (or type 'exit'): ")
    if query.lower() == "exit":
        break

    inspect_choice = input("\nDo you want to inspect retrieved context? (y/n): ").strip().lower()
    if inspect_choice == 'y':
        inspect_retrieved_docs(query)

    print("\nRunning query through RAG pipeline...")
    start_time = time.time()
    answer = rag_chain.run(query)
    end_time = time.time()

    print("\nAnswer:", answer)
    print("Time taken: {:.2f} seconds".format(end_time - start_time))
