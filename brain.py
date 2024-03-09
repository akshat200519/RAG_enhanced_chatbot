#Importing the required libraries
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from pdfminer.high_level import extract_text


def parse_pdf(pdf_documents):
    """
    Extracts text content from a list of PDF documents.

    Args:
        pdf_documents (list): A list containing PDF document objects.

    Returns:
        str: The combined text content extracted from all PDFs.
    """

    text = ""
    for pdf_document in pdf_documents:
        text += extract_text(pdf_document)  # Concatenate extracted text
    return text


def text_to_documents(text):
    """
    Splits text into smaller document chunks for processing.

    Args:
        text (str): The text content to be split.

    Returns:
        list: A list of text chunks.
    """

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def documents_to_index(documents):
    """
    Creates a vector store index from a list of text documents.

    Args:
        documents (list): A list of text documents (chunks).

    Returns:
        FAISS: A FAISS vector store object representing the document index.
    """

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=documents, embedding=embeddings)
    return vectorstore


def get_index_for_pdf(pdf_files):
    """
    Creates a FAISS vector store index from uploaded PDF documents.

    Args:
        pdf_files (list): A list of uploaded PDF file objects.

    Returns:
        FAISS: A FAISS vector store object representing the document index.
    """

    text = ""
    for pdf_file in pdf_files:
        text += parse_pdf(pdf_file)
        doc_chunks = text_to_documents(text)
    index = documents_to_index(doc_chunks)
    return index
