import streamlit as st
import os
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📚 PDF RAG Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload PDFs and get answers ONLY from your documents! 🎯</p>", unsafe_allow_html=True)

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

def check_document_relevance(query, documents):
    """Simple keyword-based relevance checking"""
    if not documents:
        return False
    
    query_words = set(query.lower().split())
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where'}
    query_words -= common_words
    
    if not query_words:
        return True
    
    # Check if any meaningful query words appear in documents
    for doc in documents:
        doc_text = doc.page_content.lower()
        if any(word in doc_text for word in query_words):
            return True
    
    return False

def validate_response(response_text, context_docs):
    """Check if response contains generic phrases or uses context"""
    if not context_docs:
        return False
    
    generic_phrases = [
        "i don't know", "i'm not sure", "cannot answer", "not mentioned", 
        "not provided", "no information", "not available", "not specified"
    ]
    
    response_lower = response_text.lower()
    if any(phrase in response_lower for phrase in generic_phrases):
        return False
    
    # Check if response uses document content
    response_words = set(response_lower.split())
    context_text = " ".join([doc.page_content.lower() for doc in context_docs])
    context_words = set(context_text.split())
    
    # Remove common words
    common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    response_words -= common_words
    context_words -= common_words
    
    if len(response_words) == 0:
        return False
    
    # Check overlap
    overlap = len(response_words.intersection(context_words))
    return overlap / len(response_words) >= 0.15

if api_key:
    llm = ChatGroq(model='Gemma2-9b-it', groq_api_key=api_key)
    
    # Session management
    session_id = st.text_input("Session ID", value='default_session')
    if 'store' not in st.session_state:
        st.session_state.store = {}
    
    # File upload
    uploaded_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)
    
    if uploaded_files:
        # Process PDFs
        with st.spinner("📄 Processing PDFs..."):
            documents = []
            for uploaded_file in uploaded_files:
                temppdf = f"./temp_{uploaded_file.name}"
                with open(temppdf, "wb") as file:
                    file.write(uploaded_file.read())
                
                loader = PyPDFLoader(temppdf)
                docs = loader.load()
                documents.extend(docs)
                os.remove(temppdf)
        
        st.success(f"✅ Processed {len(documents)} pages from {len(uploaded_files)} PDF(s)")
        
        # Create vector store
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        
        # Setup contextualization chain
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        # Enhanced system prompt for strict PDF-only responses
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "CRITICAL INSTRUCTIONS:\n"
            "1. You MUST ONLY use information from the provided context documents\n"
            "2. If the context does not contain relevant information to answer the question, "
            "you MUST respond with exactly: 'Information not available in the uploaded documents'\n"
            "3. Do NOT use your general knowledge or training data\n"
            "4. Do NOT make assumptions beyond what is explicitly stated in the context\n"
            "5. Ground your answer directly in the provided context\n"
            "6. Keep answers concise and factual (max 3-4 sentences)\n"
            "\nContext from uploaded PDF documents:\n{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session not in st.session_state.store:
                st.session_state.store[session] = ChatMessageHistory()
            return st.session_state.store[session]
        
        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        
        # Chat interface
        user_input = st.text_input("Ask your question:")
        
        if user_input:
            with st.spinner("🤔 Processing your question..."):
                docs = retriever.invoke(user_input)
                
                # Check if documents are relevant
                is_relevant = check_document_relevance(user_input, docs)
                
                if not docs or not is_relevant:
                    st.error("❌ **Assistant:** Information not available in the uploaded documents")
                else:
                    try:
                        response = conversational_rag_chain.invoke(
                            {"input": user_input},
                            config={"configurable": {"session_id": session_id}},
                        )
                        
                        answer = response.get('answer', '')
                        
                        # Validate the response
                        is_valid = validate_response(answer, docs)
                        
                        if not is_valid or "information not available" in answer.lower():
                            st.error("❌ **Assistant:** Information not available in the uploaded documents")
                        else:
                            st.success(f"✅ **Assistant:** {answer}")
                            
                            # Show source documents
                            with st.expander("📖 View Sources"):
                                for i, doc in enumerate(docs[:3]):  # Show top 3 sources
                                    st.write(f"**Source {i+1} (Page {doc.metadata.get('page', 'Unknown')}):**")
                                    st.write(f"{doc.page_content[:200]}...")
                                    st.write("---")
                    
                    except Exception as e:
                        st.error(f"❌ Error processing question: {str(e)}")
        
        history = get_session_history(session_id)
        if history.messages:
            st.markdown("### 💬 Chat History")
            for i, message in enumerate(history.messages):
                if i % 2 == 0:  
                    st.write(f"**You:** {message.content}")
                else:  
                    st.write(f"**Assistant:** {message.content}")
        
            # Clear history button
            if st.button("🗑️ Clear Chat History"):
                if session_id in st.session_state.store:
                    st.session_state.store[session_id] = ChatMessageHistory()
                    st.rerun()

else:
    st.warning("🔑 Please enter your Groq API Key in the sidebar!")


st.markdown("---")
st.markdown("### 🚀 How to use:")
st.write("1. Enter your Groq API key in the sidebar")  
st.write("2. Upload one or more PDF files")
st.write("3. Ask questions and get answers ONLY from your PDFs!")
