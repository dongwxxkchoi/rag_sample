import chromadb
from src.chromadb.process.embedding import embedding_function

def open_chroma(collection_name: str):
    chroma_client = chromadb.PersistentClient(path="./chroma")
    try:
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)
    except Exception:
        # make collection
        collection = chroma_client.get_collection(name=collection_name, embedding_function=embedding_function)

    return chroma_client