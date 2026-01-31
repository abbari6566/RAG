import os
import sys
import tempfile
import requests

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import BSHTMLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory

import numpy as np
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

OLLAMA_MODEL = "mistral"   
TEMPERATURE = 0.4

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)


PROMPT = PromptTemplate(
    template="""Context:
{context}

Question:
{question}

Answer concisely based only on the given context.
If the context does not contain the answer, say:
"I don't have enough information to answer that question."

If the question is generic (e.g., "What is an electric vehicle?"),
answer it normally.
""",
    input_variables=["context", "question"]
)

def fetch_html(url: str) -> str | None:
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"Error fetching website: {e}")
        return None

def process_website(url: str):
    html = fetch_html(url)
    if not html:
        raise ValueError("No content fetched.")

    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".html",
        encoding="utf-8",
        errors="ignore"
    ) as f:
        f.write(html)
        temp_path = f.name

    try:
        loader = BSHTMLLoader(temp_path)
        documents = loader.load()
    finally:
        os.unlink(temp_path)

    if not documents:
        return []

    print(f"\nDocuments loaded: {len(documents)}")
    print("Sample content:")
    print(documents[0].page_content[:200].replace("\n", " ") + "...")

    splitter = CharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    print(f"Text chunks created: {len(chunks)}")

    return chunks


def show_sample_embedding(chunks, embedder):
    vec = embedder.embed_query(chunks[0].page_content)
    print("\nEmbedding sample (first 10 dims):")
    print(np.array(vec[:10]))
    print("Embedding shape:", np.array(vec).shape)


def rag_pipeline(query, qa_chain, vectorstore):
    results = vectorstore.similarity_search_with_score(query, k=3)

    print("\nTop retrieved chunks:")
    context = []
    for i, (doc, dist) in enumerate(results, 1):
        sim = 1 / (1 + float(dist))
        print(f"{i}. distance={dist:.4f} | sim≈{sim:.4f}")
        print(doc.page_content[:200].replace("\n", " ") + "...\n")
        context.append(doc.page_content)

    response = qa_chain.invoke({"query": query})
    return response["result"]


if __name__ == "__main__":
    print("Fully Local RAG Pipeline (Ollama + FAISS + HF Embeddings)")

    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Local embeddings loaded")

    # Local LLM (Ollama)
    llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE
    )
    print(f"Ollama LLM loaded: {OLLAMA_MODEL}")

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    while True:
        url = input("\nEnter website URL (or 'quit'): ").strip()
        if url.lower() == "quit":
            print("Goodbye")
            break

        try:
            print("Processing website...")
            chunks = process_website(url)
            if not chunks:
                print("No usable content found.")
                continue

            show_sample_embedding(chunks, embeddings)

            print("Building FAISS vector store...")
            vectorstore = FAISS.from_documents(chunks, embeddings)

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                chain_type_kwargs={"prompt": PROMPT},
                memory=memory
            )

            print("\nRAG ready. Ask questions.")
            print("Type 'new' for a new site or 'quit' to exit.")

            while True:
                q = input("\nQuery: ").strip()
                if q.lower() == "quit":
                    sys.exit()
                if q.lower() == "new":
                    break

                answer = rag_pipeline(q, qa, vectorstore)
                print("Answer:", answer)

        except Exception as e:
            print(f"Error: {e}")
