import langchain
import langchain_community
from langchain_google_genai import GoogleGenerativeAI

from langchain.embeddings import GooglePalmEmbeddings
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
import os

load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"]=GOOGLE_API_KEY


def get_text(urls):
    
    url_reader=UnstructuredURLLoader(urls)
   
    text = url_reader.load()
    return text

def get_chunks(text):
   
    text_splitter=CharacterTextSplitter(separator="\n",
                                    chunk_size=1000,
                                    chunk_overlap=200)
    chunks=text_splitter.split_documents(text)
    return chunks

def get_vector_store(chunks):

    embeddings=GooglePalmEmbeddings()
    
    index_name="lama2web"

    vector_store=PineconeVectorStore(index_name=index_name,embedding=embeddings,pinecone_api_key=PINECONE_API_KEY)

    vector_store.from_texts(chunks,embedding=embeddings,index_name=index_name)

    return vector_store

def get_conversational_chain(vector_store):

    llm=GoogleGenerativeAI(model="models/text-bison-001",google_api_key=GOOGLE_API_KEY,temperature=0.1)
    memory=ConversationBufferMemory(memory_key="chat_history",input_key="question",output_key="answer",return_messages=True)

    conversation_chain=ConversationalRetrievalChain.from_llm(llm=llm,retriever=vector_store.as_retriever(),memory=memory)

    return conversation_chain
    
  


