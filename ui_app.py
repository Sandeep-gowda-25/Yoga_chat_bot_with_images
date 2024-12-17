import streamlit as st

from app import Application

application = Application()

st.title("Yoga chat bot!!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image" in message:
            st.image(image=message["image"])


if question := st.chat_input("Type your query here.."):
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        answer = application.get_answer(question)
        st.markdown(answer["answer"])
        if "image" in answer:
            st.image(image=answer["image"],caption=question)
            st.session_state.messages.append({"role":"assistant","content":answer["answer"],"image":answer["image"]})
        st.session_state.messages.append({"role":"assistant","content":answer["answer"]})