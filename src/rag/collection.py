from src.chromadb.connect.open import open

import chromadb
from langchain.vectorstores.chroma import Chroma
from src.utils.connect_openai import embedding_function

# open collection by the collection_name
# default path is './chroma' because we use PersistentChroma (file system based)
# later, we should change it to HTTPChroma (http server connection based)

def open_collection(collection_name):
    client = chromadb.PersistentClient(path="./chroma")

    chroma = Chroma(
            persist_directory="./chroma",
            client=client,
            collection_name=collection_name,
            embedding_function=embedding_function
        )

    return chroma