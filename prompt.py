from langchain.prompts import PromptTemplate

template = """
You are an AI assistant that answers user question regarding Yoga based on given context.
If you don't know the answer, just say I don't know, don't try to make up an answer.

CONTEXT : {context}
QUESTION : {question}

Give only the answer and nothing else.

ANSWER : 
"""
prompt = PromptTemplate(template=template,input_variables=["context","question"])