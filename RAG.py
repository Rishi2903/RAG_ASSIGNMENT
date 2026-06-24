from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

ollama_model = OllamaLLM(model="llama3.2")

print("Loading PDF documents...")
loader_paths = [
    "data/pdf-1.pdf",
    "data/pdf-2.pdf",
    "data/pdf-3.pdf",
    "data/pdf-4.pdf",
    "data/pdf-5.pdf",
]

documents = []
for path in loader_paths:
    loader = PyPDFLoader(path)
    documents.extend(loader.load())
print(f"Pages Loaded: {len(documents)}")



splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"\nTotal Chunks Created: {len(chunks)}")

for i in range(min(4, len(chunks))):
    print("\n")
    print("=" * 60)
    print(f"CHUNK {i + 1}")
    print("=" * 60)
    print(chunks[i].page_content[:5000])

print("\nGenerating embeddings:")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector = embedding_model.embed_query(chunks[0].page_content)

print("\nVector Length:")
print(len(vector))

print("\nFirst 60 numbers:")
print(vector[:60])

print("\n Creating FAISS Database...")

vector_db = FAISS.from_documents(chunks, embedding_model)

print(f"Vector Store: {vector_db.index.ntotal}")

query = input("\n Asking a question about the document:")
print(f"\nYou asked: {query}")
print("Step-1")
results = vector_db.similarity_search(query, k=3)
print("Step-2")
print("\nTop 3 Retrieved Chunks:")
for i, doc in enumerate(results):
    print("\n")
    print("=" * 60)
    print(f"RESULT {i + 1}")
    print("=" * 60)
    print(doc.page_content[:1000])

    context = "\n\n".join([doc.page_content for doc in results])
    
    prompt = f"""
Answer only from the context.

Context:
{context}
Question:
{query}

Answer:
"""
    response =  ollama_model.invoke(prompt)
print("\n Answer:")
print(response)

#Without RAG
normal_response = ollama_model.invoke(query)
print("\n")
print("=" * 60)
print("Without RAG")
print("=" * 60)
print(normal_response)