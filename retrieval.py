import openai
import pandas as pd
import os

from langchain.prompts import PromptTemplate
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Chroma's client library for Python
import chromadb

from prepare_chroma import open_chroma, load_data

# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

################ environment settings ################
######################################################

EMBEDDING_MODEL = "text-embedding-ada-002"
api_key = os.environ.get('OPENAI_API_KEY')

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print ("OPENAI_API_KEY is ready")
else:
    print ("OPENAI_API_KEY environment variable not found")

################ Chroma DB ################
###########################################

def query_collection(collection: Chroma, query, max_results, dataframe):
    # k: top-k, filter: metadata
    results = collection.similarity_search(query)
    # results = collection.query(query_texts=query, n_results=max_results, include=['distances']) 
    df = pd.DataFrame({
                'id':results['ids'][0], 
                'score':results['distances'][0],
                'title': dataframe[dataframe.vector_id.isin(results['ids'][0])]['title'],
                'content': dataframe[dataframe.vector_id.isin(results['ids'][0])]['text'],
                })
    
    return df

def main():
    # load chroma db
    wikipedia_content_collection, wikipedia_title_chroma = open_chroma()
    df = load_data()
    
    # open chroma client
    chroma_client = chromadb.PersistentClient()
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-ada-002",
    )

    # embedding_function = OpenAIEmbeddingFunction(api_key=os.environ.get('OPENAI_API_KEY'), model_name=EMBEDDING_MODEL)
    
    #### pass over chroma to langchain ####
    wikipedia_content_chroma = Chroma(
        persist_directory="./chroma",
        client=chroma_client,
        collection_name="wikipedia_content",
        embedding_function=embedding_function
    )

    wikipedia_title_chroma = Chroma(
        persist_directory="./chroma",
        client=chroma_client,
        collection_name="wikipedia_title",
        embedding_function=embedding_function
    )
    wikipedia_content_chroma.similarity_search
    query = "What did the president say about Ketanji Brown Jackson"
    query_collection(wikipedia_content_chroma, query, 10, df)
    
    #### query to chroma db ####

    

if __name__ == "__main__":
    main()