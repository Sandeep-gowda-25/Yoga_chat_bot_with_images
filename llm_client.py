from huggingface_hub import InferenceClient,login
import requests
from together import Together
from dotenv import load_dotenv
load_dotenv('.env')
import os
from sentence_transformers import SentenceTransformer
from langchain.llms.base import LLM

class LLMOperations:
    '''
    Thic calls initializes LLM client with required configurations and
    gives methods to get response and also to get embededdings of given text input
    '''

    def __init__(self,*args,**kwargs):
        self.hf_token = os.getenv("HUGGING_FACE_TOKEN")
        self.hf_client = InferenceClient(model="meta-llama/Llama-3.2-1B-Instruct")
        self.together_client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL")
        self.deployment_name = os.getenv("DEPLOYMENT_NAME")
        system_message = "You are an AI assistant that answers questions regarding Yoga based on given context."
        self.messages = [{"role":"system","content":system_message}]
        self.hf_embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.hf_embedding_headers = {"Authorization": f"Bearer {self.hf_token}"}
        login(self.hf_token)
        self.supabase_bucket_url = os.getenv("SUPABASE_BUCKET_URL")


    def get_response(self,query:str,context:str):
        messages = [{"role":"user","content":"""Generate answer only as per the given contents, don't make up an answer on your own
        CONTENTS : {context}"""},
        {"role":"user","content":"QUESTION : {query}"}]
        self.messages.extend(messages)
        chat_completion = self.hf_client.chat_completion(
            model="meta-llama/Llama-3.2-1B-Instruct", 
            messages=self.messages, 
            max_tokens=500
        )
        output = chat_completion.choices[0].message.content
        return output
    
    def get_image_to_text(self,image_name:str):
        try:
            response = self.together_client.chat.completions.create(
                model="meta-llama/Llama-Vision-Free",
                messages=[
                    {
                        "role": "user",
                        "content": [{"type": "text","text": "Give brief details about this yoga posture?"},
                                    {"type": "image_url",
                                        "image_url": {
                                            "url": f"{self.supabase_bucket_url}/{image_name}"
                                        }
                                    }]
                    }
                ],
                max_tokens=512,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1,
                stop=["<|eot_id|>","<|eom_id|>"],
                stream=True,
            )
            output = ''
            for token in response:
                if hasattr(token, 'choices'):
                    try:
                        output = output+ (token.choices[0].delta.content)#, end='', flush=True)
                    except Exception as e:
                        pass
            return output
        except Exception as e:
            print(f"Error occured reading image - {image_name}")  

    def get_text_embeddings(self,texts):
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        return embeddings

class Llama(LLM):

    @property
    def _llm_type(self)->str:
        return "Llama3.2 1B"

    def _call(self,prompt:str,**kwargs)->str:
        # system_message = "You are an AI assistant that answers questions regarding Yoga based on given context."
        # messages = [{"role":"system","content":system_message},
        # {"role":"user","content":prompt}]
        # messages = [{"role":"user","content":"""Generate answer only as per the given contents, don't make up an answer on your own
        # CONTENTS : {context}"""},
        # {"role":"user","content":"QUESTION : {prompt}"}]
        # self.messages.extend(messages)
        hf_client = InferenceClient(model="meta-llama/Llama-3.2-1B-Instruct")
        chat_completion = hf_client.chat_completion(
            model="meta-llama/Llama-3.2-1B-Instruct", 
            messages=[{"role":"user","content":prompt}], 
            max_tokens=500,
            temperature=0.3
        )
        output = chat_completion.choices[0].message.content
        return output
