# ingest_csv.py
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader

csv_file = "C:\\Users\\XC\\Downloads\\FYP_latest - Copy\\FYP_latest - Copy\\sales_data_sample.csv"
loader = CSVLoader(csv_file)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_docs = splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
store = PineconeVectorStore(
    index_name="fyp",
    embedding=embeddings
)
store.add_documents(splitted_docs)
