import streamlit as st
from openai import OpenAI
from phlex import *



# Show title and description.
st.title("💬 PHLEX Chatbot")
st.write("Welcome to Philippine Legal Expert (PHLEX). \n")
st.write( "Type in your legal question and we'll provide you the answer. \n")
st.write("YES, it's that simple!!!")

# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.secrets["OPENAI_API_KEY"]
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Initialize OpenAI client and RagBot
    client = OpenAIChatLLM(api_key=openai_api_key)
    rag_bot = RagBot(llm=client, verbose=True)  # Set verbose to True if you want detailed logs

    # Create a session state variable to store the chat messages and bot responses.
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.responses = []

    # Display the existing chat messages via `st.chat_message`.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input field for the user
    if prompt := st.chat_input("What is up, my bro?"):
        # Store and display the current prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Use RagBot to generate a response based on the Pinecone query
        response = rag_bot.run(prompt)

        # Store the response in session state and display it
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
