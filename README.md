# Ollama Chatbot with Streamlit

## Overview

This project is a chatbot interface built using Streamlit and Ollama's language models. It allows users to interact with an AI-powered chatbot, dynamically selecting models and customizing system prompts for tailored responses.

## Features

- **Dynamic Model Selection**: Users can choose from available models retrieved via `ollama.list()`.
- **System Prompt Customization**: Modify the system's behavior and response style via a sidebar input.
- **Chat History**: Maintains a session-based history of messages for a continuous conversation flow.
- **Streamlit UI**: A simple and interactive chat interface using Streamlit's `chat_message` and `chat_input` components.

## Installation

Ensure you have Python installed, then clone this repository and install the dependencies:

```sh
pip install -r requirements.txt
```

## Running the App

Run the Streamlit application using the following command:

```sh
streamlit run app.py
```

## Dependencies

The required dependencies are listed in `requirements.txt`. They include:

- `streamlit`
- `ollama`

## Usage

1. Start the application with `streamlit run app.py`.
2. Select a model from the sidebar dropdown.
3. Optionally customize the system prompt.
4. Enter a message in the chat input field.
5. Receive AI-generated responses while maintaining conversation history.

## File Structure

- `app.py`: Main Streamlit application script.
- `requirements.txt`: List of required Python packages.

## License

This project is open-source and available for modification and use under the appropriate license.

## Author

LDSPrgrm
