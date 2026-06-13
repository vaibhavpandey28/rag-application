
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from app.rag import RAGService
from pydantic import  SecretStr
import os   
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = SecretStr(os.getenv("OPENAI_API_KEY", "sk-not-needed"))



embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) 




HF_TOKEN = SecretStr(os.getenv("HUGGINGFACEHUB_API_TOKEN" ,"default"))

llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-R1",
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)


# Chat Model
# llm = ChatOpenAI(
#     model="google/gemma-4-e2b",
#     base_url="http://127.0.0.1:2402/v1",
#     api_key=API_KEY,
#     temperature=0,
# )

rag = RAGService(
    embeddings=embeddings,
    llm=llm,
)
