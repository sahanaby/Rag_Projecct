import streamlit as st
from rag_pipeline import load_pdf, split_documents, create_embeddings, create_vector_store, create_qa_chain


st.title("Document Question Answering System")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    docs = load_pdf("temp.pdf")

    chunks = split_documents(docs)

    embeddings = create_embeddings()

    vector_db = create_vector_store(chunks, embeddings)

    qa_chain = create_qa_chain(vector_db)

    question = st.text_input("Ask your question")

    if question:

        with st.spinner("Searching answer..."):

            result = qa_chain({"query": question})

            st.subheader("Answer")
            st.write(result["result"])

            st.subheader("Source Chunks")

            for doc in result["source_documents"]:
                st.write(doc.page_content)