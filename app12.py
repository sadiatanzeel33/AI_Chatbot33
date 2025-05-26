from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser

import streamlit as st

# 🛠 Streamlit Page Config
st.set_page_config(page_title="AI Text Assistant", layout="centered")
st.title("🤖 QueryMind AI Chatbot")
st.markdown("Welcome! I can chat with you in **English** and **Urdu**. Try asking me something!")

# 🔐 Google API Key (Now securely accessed from Streamlit secrets)
# REMOVE the hardcoded API key line: api_key = "AIzaSyC2-R5FykvtAjUc9g3SxwM_oiWcpJ9Ax7E"
# Instead, access it from Streamlit secrets
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Key not found in Streamlit secrets. Please add it to your `secrets.toml` file or Streamlit Cloud secrets.")
    st.stop() # Stop the app if API key is not found

# 📜 Prompt Template
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are a helpful, friendly AI assistant. If the user asks to explain or respond in Urdu, do so. Otherwise, use English."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{question}")
])

# 💬 Chat History
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# 🤖 Gemini Model
model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)

# 🔗 Chain Logic
chain = prompt | model | StrOutputParser()

# 🧠 Chain with Message History
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# 💬 Streamlit Chat Interface
if user_input := st.chat_input("Ask me anything..."):
    # Show user's message
    st.chat_message("user").write(user_input)

    # Call the LangChain chain
    response = chain_with_history.invoke(
        {"question": user_input},
        config={"configurable": {"session_id": "chat-session-001"}}
    )

    # Show assistant's reply
    st.chat_message("assistant").write(response)