import openai
import os

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

EMBEDDING_MODEL = "text-embedding-ada-002"
embedding_function = None

if os.getenv("OPENAI_API_KEY") is not None:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    print("OPENAI_API_KEY is ready")
    embedding_function = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=EMBEDDING_MODEL)
    
else:
    raise Exception("OPENAI_API_KEY environment variable not found")
