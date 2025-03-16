import streamlit as st
import ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="💬 Ollama Chatbot")

def get_available_models():
    """Retrieve available models from Ollama."""
    models = ollama.list()
    return [model["model"] for model in models.get("models", [])]

def chat_with_ollama(user_input, history, system_prompt, model):
    """Send user input and chat history to Ollama and return response."""
    messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [{"role": "user", "content": user_input}]
    
    response = ollama.chat(
        model=model,
        messages=messages
    )
    return response["messages"]["content"]

@st.cache_resource
def load_pdf(uploaded_file):
    """Load and index the uploaded PDF."""
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loaders = [PyPDFLoader("uploaded.pdf")]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    index = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=text_splitter
    ).from_loaders(loaders)
    
    return index

def main():
    st.title("💬 Ollama Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.sidebar.header("Model Settings")
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox("Select Model", available_models, index=0)

    # system_prompt = st.sidebar.text_area(
    #     "System Prompt", 
    #     "You are an AI that responds naturally in casual conversation. Keep replies short, engaging, and flowing like a real chat."
    # )

    # PDF Upload
    uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for Q&A", type="pdf")

    if uploaded_pdf:
        index = load_pdf(uploaded_pdf)

        llm = OllamaLLM(model=selected_model)  # Use Ollama with LangChain

        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=index.vectorstore.as_retriever(),
            input_key="question"
        )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Ask me anything...")
    if user_input:
        st.chat_message("user").markdown(user_input)

        # Default Ollama Chat Response
        # response = chat_with_ollama(user_input, st.session_state.messages, system_prompt, selected_model)

        # If PDF uploaded, run document retrieval QA
        if uploaded_pdf:
            response = chain.run(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
