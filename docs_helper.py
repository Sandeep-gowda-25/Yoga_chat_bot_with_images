import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
import nltk
from typing import List
from dotenv import load_dotenv
load_dotenv(".env")
import os
from supabase import create_client

def load_pdf_data():
    file_path = "HealthFitness_YOGA_Manual.pdf"
    loader = pypdf.PdfReader(file_path)

    document_contents=[]
    for page in loader.pages:
        page_text = page.extract_text()
        chunks = chunk_data(page_text)
        document_contents=document_contents+chunks
    return document_contents

def load_image_data():
    from PIL import Image
    from io import BytesIO
    file_path = "HealthFitness_YOGA_Manual.pdf"
    loader = pypdf.PdfReader(file_path)
    image_count = 0
    for page in loader.pages:
        for image in page.images:
            if len(image.data) >= 1000:##condition to avoid junk or false images of size less than 1kb
                name = f"images/image_{image_count}.png"
                img = image.image
                img.save(name)
                image_count = image_count+1
    print("Reading images frm pdf and saving to folder is completed")

def chunk_data(text_input:str):
    splitter = RecursiveCharacterTextSplitter(separators=['\n'],chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(text_input)
    return chunks

def process_input(text_input:str):
    word_list = tokenization(text_input)
    word_list = stop_words_removal(word_list)
    word_list = lematization(word_list)

    cleaned_text = ' '.join(word_list)
    return cleaned_text


def tokenization(text_input:str):
    nltk.download('punkt',quiet=True)
    from nltk.tokenize import word_tokenize
    word_list = word_tokenize(text_input)
    return word_list

def stop_words_removal(word_list:List[str]):
    nltk.download('stopwords',quiet=True)
    from nltk.corpus import stopwords
    stopwords = stopwords.words('english')
    for i,word in enumerate(word_list):
        if word.lower() in stopwords:
            word_list.pop(i)
    return word_list
    
def lematization(word_list:List[str]):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(word) for word in word_list]

class SupaBaseOperations:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.supabase_client = create_client(self.supabase_url,self.supabase_key)
        self.bucket_name = "yoga_images"


    def upload_files_to_supabase(self):
        for filename in os.listdir("images"):
            file_path = os.path.join("images", filename)
            with open(file_path,"rb") as image_file:
                self.supabase_client.storage.from_(self.bucket_name).upload(
                    file=image_file,
                    path=filename
                )
