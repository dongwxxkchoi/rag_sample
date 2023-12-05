import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from langchain.schema.document import Document

from langchain.vectorstores.chroma import Chroma
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import chromadb

from rag.collection import open_collection
from utils.connect_openai import EMBEDDING_MODEL, embedding_function



# Retriever object per one collection
# When it is created, all the settings are conducted 
# based on the choices based on the front-end's setting value

class Retriever:
    # Should change PersistentClient to HTTPClient later
    # PersistentClient is not for production
    def __init__(self, collection_name: str, search_type: str = "mmr", **search_kwargs):
        self.embedding_function: OpenAIEmbeddingFunction = embedding_function
        self.collection_name: str = collection_name

        self.chroma: Chroma = open_collection()

        if search_kwargs is not None:
            self.retriever = self.chroma.as_retriever(search_type=search_type, 
                                                    search_kwargs=search_kwargs)
        else:
            self.retriever = self.chroma.as_retriever(search_type=search_type)

    def retrieve_setting(self):
        tool = create_retriever_tool(
            self.retriever,
            "search_IR",
            "Searches and returns documents regarding the IR reports.",
        )

        tools = [tool]

    def retrieve(self, query: str) -> List[Document]:
        
        docs: List[Document] = self.retriever.get_relevant_documents(query)
        return docs
    
