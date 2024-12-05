from handle_vectors import pinecone_retriever
from prompt import prompt
from llm_client import Llama
from langchain.chains import RetrievalQA

class Application:
    def __init__(self):
        self.llm = Llama()
        retriever = pinecone_retriever()
        chain_type_kwargs = {"prompt":prompt}
        self.qa = RetrievalQA.from_chain_type(llm=self.llm,
                                        chain_type = "stuff",
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs
                                        )
        self.conv_history = []

    def get_answer(self,query):
        self.conv_history.append({"question":query})
        query = self.get_updated_question()
        response = self.qa({"query":query})
        result = response["result"]
        from IPython.display import Image,display
        sources = response["source_documents"]
        print(result)
        if "image" in str(sources) and ".png" in str(sources):
            print("Image below might give you more details/posture for your query")
            for source in sources:
                if "image" in source.metadata.keys():
                    display(Image(filename=f"images/{source.metadata["image"]}"))
                    break
        self.conv_history.append({"answer":result})

    def get_updated_question(self):

        prompt = f"""You're an conversation agent.
        Based on the given conversation history, check if the last question is depended on coversation history and update it according as required.
        If the last question is not related to earlier conversation, then give the same question updated.
        Just return the final question, no explainations should be given. Don't make up any thing el

        CONVERSATION HISTORY : {str(self.conv_history)}
        ==========
        FINAL QUESTION :"""
        updated_question = self.llm.invoke(prompt)

        return updated_question


