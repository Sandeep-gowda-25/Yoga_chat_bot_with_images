# Yoga_chat_bot_with_images
## Chat bot application developed with multimodal capability of answering Yoga questions with image in responses in applicable cases.

- This Application is desigened to read images along with text contents from Yoga related pdf, and then images with be processed with Image-to-text LMM model.
- Text generated from LMM will be used to answer the question along with original text contents as well.
- Text contents and Image texts contents will be embedded and stored in Vector Database.
- Image names will be stored as a metadata field of thier respecitive vector
- On receiving the question, application will do to similarity check on the source vector store and retrieve top 5 mathces,
- Answer will be created based on the retrieved contents and will same will be returned to user.
- Additionally, If retieved content has image metadata, then that too will be visible to user.

## Tech Stack Used:
- Python
- Llama-3.2-1B-Instruct(Text Generation LLM) - via Hugging face interference api
- Llama-Vision-Free(Image-to-text LMM) - via Together api
- Sentence Transformers/all-MiniLM-L6-v2(Embeddding model)
- Supabase(Storage account for Images)
- Pinecone(Vector DB)

#### Yoga mutlimodal chatbot.ipynb is the controlling unit of application flow

#### Interactive UI interface using streamlit (streamlit run ui_app.py)

## Sample Response:
Response along with image and context based answering for follow-up
![image](https://github.com/user-attachments/assets/c9d2d6dd-79ae-4a8c-a000-3ed09255eebc)



