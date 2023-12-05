import openai
import pandas as pd
import os

import chromadb

from prepare_chroma_pdf import open_chroma
# from prepare_chroma_sm import open_chroma

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

from langchain.prompts import PromptTemplate
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents import AgentExecutor

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

################ environment settings (for query embedding) ################
############################################################################

################ Chroma DB ################
###########################################

from retriever import Retriever

class RAG:
    def __init__(self):
        self.retriever = None

    def make_retriever(self, collection_name, search_type, **search_kwargs):
        self.retriever = Retriever(collection_name=collection_name, 
                                   search_type=search_type,
                                   search_kwargs=search_kwargs)

    def send_query(self, query):
        docs = self.retriever.retrieve(query)

    