Enhanced Q&A Chatbot with Groq
Project Overview
This project is an end-to-end Generative AI application featuring a Q&A chatbot. It is designed to demonstrate the power of real-time, low-latency AI responses by integrating the Groq API. The application uses a robust RAG (Retrieval-Augmented Generation) pipeline and a conversational memory agent to provide intelligent and context-aware answers.

Key Features
Real-time Responses: Leverages the Groq API to deliver extremely fast, low-latency conversational AI responses.

Conversational Memory: Implements a state-management system to maintain a persistent chat history for a more natural conversation flow.

RAG Pipeline: Integrates with a vector database to enable semantic search, summarization, and classification, enhancing the chatbot's knowledge beyond its base model.

Dynamic UI: Built with the Streamlit framework for a responsive and intuitive web-based user interface.

Scalable Architecture: Designed with containerization in mind using Docker, ensuring portability and easy deployment.

Secure API Handling: Uses environment variables (.env file) to securely manage API keys, following best practices.

Technologies Used
Python: The core programming language.

Groq API: For high-speed LLM inference.

LangChain: Framework for building the intelligent agent, managing conversation chains, and implementing RAG.

Streamlit: For creating the interactive user interface.

Vector Databases: For building RAG pipelines.

python-dotenv: For secure environment variable management.
