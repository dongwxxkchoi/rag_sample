import pandas as pd
import os
import wget
import zipfile
from ast import literal_eval

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain.vectorstores.chroma import Chroma

EMBEDDING_MODEL = "text-embedding-ada-002"

def load_data():
    embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'

    if 'vector_database_wikipedia_articles_embedded.csv' in os.listdir('data'):
        print("Data already exists")
    else:
        print("Download data...")
        wget.download(embeddings_url)

        with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip","r") as zip_ref:
            zip_ref.extractall("data")


def process_data():
    print("Load data...")
    article_df = pd.read_csv('data/vector_database_wikipedia_articles_embedded.csv').loc[:100]
    print("Data loaded")

    print("Process df...")
    # Read vectors from strings back into a list
    article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
    article_df['content_vector'] = article_df.content_vector.apply(literal_eval)
    
    # Set vector_id to be a string
    article_df['vector_id'] = article_df['vector_id'].apply(str)

    print("Process finished")

    return article_df

def df_to_chroma(df: pd.DataFrame, name: str):
    # PersistentClient는 chroma db를 파일에 저장함
    # EphmeralClient - 메모리에 저장
    # HttpClient - 네트워크 통해 접속 (실제 서비스에 권장)
    print("Insert df to chroma...")
    chroma_client = chromadb.PersistentClient(path="./chroma/")
    embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)
    
    if 'content' in name:
        collection = chroma_client.create_collection(name=name, embedding_function=embedding_function)    
        collection.add(
            ids=df.vector_id.tolist(),
            embeddings=df.content_vector.tolist(),
            documents=df.text.tolist()
        )
    
    elif 'title' in name:
        collection = chroma_client.create_collection(name='wikipedia_title', embedding_function=embedding_function)
        collection.add(
            ids=df.vector_id.tolist(),
            embeddings=df.title_vector.tolist(),
            documents=df.title.tolist()
        )    

    print("Insert completed")

    return collection

def open_chroma():
    chroma_client = chromadb.PersistentClient(path="./chroma/")
    embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)
    try:
        wikipedia_content_collection = chroma_client.get_collection(name='wikipedia_content', embedding_function=embedding_function)
        print("open content chroma db")
    except:
        load_data()
        article_df = process_data()
        wikipedia_content_collection = df_to_chroma(article_df, "wikipedia_content")
    try:
        wikipedia_title_collection = chroma_client.get_collection(name='wikipedia_content', embedding_function=embedding_function)
        print("open title chroma db")
    except:
        if not article_df:
            load_data()
            article_df = process_data()
        wikipedia_title_collection = df_to_chroma(article_df, "wikipedia_title")
    
    return wikipedia_content_collection, wikipedia_title_collection