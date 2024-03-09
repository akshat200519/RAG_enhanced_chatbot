#Importing the required libraries
import streamlit as st
from brain import get_index_for_pdf
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from html_Template import css, bot_template, user_template
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")


def conversation_chain(index):
    """
    Creates a ConversationalRetrievalChain object for the chatbot interaction.

    Args:
        index (object): The index object retrieved from processing uploaded PDFs.

    Returns:
        ConversationalRetrievalChain: The conversational chain object.
    """

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=model, retriever=index.as_retriever(), memory=memory, tokenizer=tokenizer
    )
    return conversation_chain


def handle_userinput(user_question):
    """
    Handles user input, updates chat history, and displays conversation messages.

    Args:
        user_question (str): The user's question entered in the text input field.
    """

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    """
    The main function that runs the chatbot application.
    """

    load_dotenv()

    prompt = """You are provided the task to be a helpful assistant who answers questions according to the multiple contexts
                 given by the user to you. Try to keep your answers short and to the point."""

    st.set_page_config(page_title="RAG Enhanced Chatbot")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("RAG Enhanced Chatbot")
    st.subheader("Upload PDF")

    pdf_files = st.file_uploader("Upload", accept_multiple_files=True, type="pdf")

    if pdf_files:
        if st.button("Process"):
            index = get_index_for_pdf(pdf_files)
            st.write("Indexing complete")

            st.session_state.conversation = conversation_chain(index)

    user_question = prompt + st.text_input("Ask question about your pdf:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "Hello human"), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
