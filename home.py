import streamlit as st
from retrieval import chain_load_db

st.title("Wiki search chat bot")
st.caption("streamlit chatbot with wikipedia data in chroma DB")


def load_client():
    collection, client = chain_load_db()
    st.session_state['client'] = client
    st.session_state['collection'] = collection

if "messages" not in st.session_state:
    load_client()
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    query = st.session_state.messages[-1]['content']
    response = st.session_state['collection'].similarity_search(query=query, k=1)
    # st.session_state.messages.append({"role": "assistant", "content": response})
    # response = client.chat.completions.create(model="gpt-3.5-turbo", messages=st.session_state.messages)
    msg = response[0].page_content
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)