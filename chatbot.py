import streamlit as st
import sys
import PyPDF2
import langchain

#insert page header
st.header("My First Chat Bot")

#insert sidebar
with st.sidebar:
    st.title("Your Documents")
    file=st.file_uploader("Upload your pdf file and start asking questions", type="pdf")

#extract text
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

if file is not None:
    pdf_reader=PdfReader(file) #store the entire read content 
    text=""
    #store each page in pdf_reader
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:  # Handle case where extract_text returns None
            text += content
            
    # Display extracted text (optional for debugging)
    #st.subheader("Extracted Text from PDF")
    #st.write(text)

#break into chunks
    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""], #This will try to split text first by paragraphs (\n\n), then lines (\n), then words (space), then characters.
    chunk_size=1000,
    chunk_overlap=150,
    length_function=len
)
    chunks=text_splitter.split_text(text)
    #st.write(chunks)

#create embeddings and vector store
    #generate embeddings for each chunk
    OPENAI_API_KEY=""
    embeddings=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    #create FAISS vector store : DB will have both chunks and corresponding embeddings
    st.write(f"Number of chunks: {len(chunks)}")  # For debugging
    if len(chunks) == 0:
        st.error("No content was extracted from the PDF. Please upload a valid file.")
    else:
        vector_store = FAISS.from_texts(chunks, embeddings)

#get user question
    user_question=st.text_input("Ask anything")

#do similarity search
#question is converted to chunks internally, then embeddings and then matched with file chunks in vector store. Match results are returned
    if user_question:
        match=vector_store.similarity_search(user_question) #match will have list of all CHUNKS that have matched
        #st.write(match) #will get multiple chunks that has answer to the question

        
#define llm model which will generate response based on obtained matched chunks
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.2, max_tokens=1000, model_name='gpt-3.5-turbo')

#output the result - chaining
#chain -> take question, get relevant document, pass it to llm, generate output
        chain=load_qa_chain(llm, chain_type="stuff") #stuff the results in bucket and send to llm
        response=chain.run(question=user_question, input_documents=match)
        st.write(response)


