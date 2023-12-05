from langchain.document_loaders import PyPDFLoader
import os
import itertools

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.vectorstores.chroma import Chroma

EMBEDDING_MODEL = "text-embedding-ada-002"

def load_data():

    page_list = []
    for file in os.listdir('data/IR/SM'):
        if file.endswith(".pdf"):
            loader = PyPDFLoader("data/IR/SM/"+file)
            pages = loader.load_and_split()
    
        page_list.append(pages)

    page_list = list(itertools.chain(*page_list))
    return page_list

def pdf_to_chroma(page_list):
    print("Insert df to chroma...")
    chroma_client = chromadb.PersistentClient(path="./chroma/")
    embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)
    try:
        collection = chroma_client.create_collection(name="IR_SM_collection", embedding_function=embedding_function)       
    except:
        collection = chroma_client.get_collection(name="IR_SM_collection", embedding_function=embedding_function)       

    collection.add(
            ids=[str(i) for i in range(len(page_list))],
            documents=[page.page_content for page in page_list],
            metadatas=[page.metadata for page in page_list]
    )

    return chroma_client, collection
    
def open_chroma():
    chroma_client = chromadb.PersistentClient(path="./chroma/")
    embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)
    try:
        IR_collection = chroma_client.get_collection(name='IR_SM_collection', embedding_function=embedding_function)
    except:
        page_list = load_data()
        IR_collection = pdf_to_chroma(page_list)
    
    return chroma_client

# if __name__=="__main__":
#     open_chroma()