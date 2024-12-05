from llm_client import LLMOperations
from pinecone import Pinecone,ServerlessSpec
from dotenv import load_dotenv
load_dotenv('.env')
import os

def create_vectors_processed(chunks):
    llm = LLMOperations()
    embeddings = llm.get_text_embeddings(chunks)
    vectors=[]
    for i,chunk in enumerate(chunks):
        vectors.append({
            "id":str(i),
            "values":embeddings[i],
            "metadata":{'text': chunk}
        })
    return vectors

def create_vectors_for_image_data(chunks,seed=0):
    images = [list(chunk.keys())[0] for chunk in chunks]
    texts = [list(chunk.values())[0] for chunk in chunks]
    llm = LLMOperations()
    embeddings = llm.get_text_embeddings(texts)
    vectors=[]
    for i,chunk in enumerate(texts):
        vectors.append({
            "id":str(i+seed),
            "values":embeddings[i],
            "metadata":{
                    'text': texts[i],
                    'image':images[i]
                }
        })
    return vectors

def store_vectors_processed(vectors):
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    index_name = "yogasanas"
    if index_name not in pc.list_indexes().names():
        pc.create_index(name=index_name,dimension=384,
            spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ))
    
    index = pc.Index(index_name)
    index.upsert(vectors=vectors,namespace="texts")
    return index

def delete_index():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "mahabharata"
    pc.delete_index(index_name)

def pinecone_retriever():
    from langchain_pinecone import Pinecone
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_store = Pinecone.from_existing_index(
        index_name="yogasanas",
        embedding=embeddings,
        namespace="texts",
        )
    retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})
    return retriever