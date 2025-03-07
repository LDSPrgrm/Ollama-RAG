import streamlit as st
import ollama

st.set_page_config(page_title="Ollama Chatbot")

def get_available_models():
    """Retrieve available models from Ollama."""
    models = ollama.list()
    return [model["model"] for model in models.get("models", [])]

def chat_with_ollama(user_input, history, system_prompt, model, temperature=0.7, top_k=40, top_p=0.9, mirostat=0, num_ctx=2048, repeat_penalty=1.1, num_predict=200):
    """Send user input and chat history to Ollama and return response with advanced options."""
    messages = [
        {"role": "system", "content": system_prompt}
    ] + history + [{"role": "user", "content": user_input}]
    
    response = ollama.chat(
        model=model,
        messages=messages
    )
    return response["message"]["content"]

def main():
    st.title("💬 Ollama Chatbot")
    
    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar parameters
    st.sidebar.header("Model Settings")
    available_models = get_available_models()
    selected_model = st.sidebar.selectbox("Select Model", available_models, index=0)
    system_prompt = st.sidebar.text_area("System Prompt", "You are an AI that responds like a human in casual conversation. Keep replies short, natural, and engaging. Use a mix of sentence lengths, occasional informal language, and emotional awareness. Show empathy, ask follow-up questions, and avoid sounding robotic. Keep it flowing like a real chat—concise but engaging.")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # User input
    user_input = st.chat_input("Ask me anything...")
    if user_input:
        # Display user message
        st.chat_message("user").markdown(user_input)
        
        # Get response from Ollama
        response = chat_with_ollama(user_input, st.session_state.messages, system_prompt, selected_model)
        
        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Update session state
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()