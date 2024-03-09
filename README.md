# RAG_enhanced_chatbot


### Description:

This repository provides a Python implementation of a Retrieval-Augmented Generation (RAG) enhanced chatbot that can answer questions based on uploaded PDFs. It leverages pre-trained language models and vector stores for efficient information retrieval and response generation.

- - - -

### Key Features:

* Uploads and processes multiple PDFs.
* Creates a FAISS vector store index from extracted text content.
* Enables users to ask questions about the PDFs.
* Generates informative responses using a pre-trained conversational language model.
* Streamlit framework for a user-friendly web interface (requires additional setup).

- - - -

### Installation and Usage:

1. **Clone this Repository:**
```bash
   git clone https://github.com/akshat200519/RAG_enhanced_chatbot.git
```
2. **Install required dependencies:**
```bash
   pip install -r requirements.txt
```
3. **Run app.py file:**
```bash
   streamlit run (file_path)app.py
``` 

- - - -

### How it works:

1. The user uploads one or more PDF documents.
2. Text is extracted from each uploaded PDF using `pdfminer`.
3. The extracted text is split into smaller chunks for efficient processing.
4. A vector store index is created using `FAISS` and pre-trained sentence embeddings `(HuggingFaceInstructEmbeddings)`.
5. The user interacts with the chatbot by asking questions.
6. The user's question is processed and used to retrieve relevant document chunks from the vector store.
7. A pre-trained conversational language model `(google/gemma-2b)` is used to generate a response based on the retrieved information.

- - - -

### Additional Notes:

* This is a basic implementation and can be further customized.
* The accuracy of the answers depends on the quality and relevance of the uploaded PDF documents.
* Refer to the code comments for detailed explanations of each function.

- - - -

### License:

This project is licensed under the MIT License: https://opensource.org/licenses/MIT.

