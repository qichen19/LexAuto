from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import os
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

pinecone.init(
    api_key=PINECONE_API_KEY, environment="gcp-starter"
)

if __name__ == "__main__":
    # loader = TextLoader(
    #     "/Users/qichen/Documents/Volunteer/LEXIFY/AI Projects/DocReader/langchain_wikipedia.txt"
    # )
    loader = PyPDFLoader(
        file_path="/Users/qichen/Documents/Volunteer/LEXIFY/AI Projects/DocReader/langchain_research.pdf")
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    texts = text_splitter.split_documents(document)
    print(len(texts))
    embeddings = OpenAIEmbeddings(
        openai_api_key="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"
    )
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="langchain-doc-index")
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key="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"), chain_type="stuff", retriever=docsearch.as_retriever())
    query = "what is a vector store? "
    result = qa({"query": query})
    print(result)
