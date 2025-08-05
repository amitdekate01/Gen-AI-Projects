import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

## Langsmith
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'QA Chatbot with Groq'

groq_api_key = os.getenv('GROQ_API_KEY')

## Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

def generate_response_with_memory(question, api_key, model_name, temperature, max_tokens):
    
    llm = ChatGroq(model=model_name, groq_api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    
    conversation = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=True 
    )
    
    # Invoke the conversation chain
    answer = conversation.predict(input=question)
    
    return answer

## Title of the app
st.title("Enhanced Q&A Chatbot with Groq")

## Sidebar for settings
st.sidebar.title("Settings")
api_key_input = st.sidebar.text_input("Enter your Groq API Key:", type="password", value=groq_api_key)

## Drop down to select various Groq Models
select_model = st.sidebar.selectbox("Select a Groq Open Source Model", ["gemma2-9b-it", "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "meta-llama/llama-guard-4-12b"])

## Adjust response parameter
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=2000, value=500)

## Main interface for user input
st.write("Go ahead and ask any question!!!")

# Display the chat history
for message in st.session_state.chat_history:
    st.write(f"**{message['role']}**: {message['content']}")

user_input = st.text_input("You:")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Use the new function with memory
    try:
        response = generate_response_with_memory(user_input, api_key_input, select_model, temperature, max_tokens)
        
        # Add the AI's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display the new response
        st.write(f"**Assistant**: {response}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error. Please try again."})
        st.write("Sorry, I encountered an error. Please try again.")

else:
    st.write("Please provide the query!!")