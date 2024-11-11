from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from dotenv import dotenv_values
import os

    
config = dotenv_values(".env")

MONGO_USER = config["MONGO_USER"]
MONGO_PASS = config["MONGO_PASS"]
mongo_connection_string = f"mongodb+srv://{MONGO_USER}:{MONGO_PASS}@ai-cluster.ffx00.mongodb.net/D01?retryWrites=true&w=majority&appName=AI-Cluster"
mongo_client = MongoClient(mongo_connection_string)
collection = mongo_client["D01"]["pdf-test"]

dir = "./pdfs"

using_zhipu = True

def main():
    print("we're using zhipu:", using_zhipu)

    for i in range(96, 101):
        file = f"pages_{i}.pdf"
        file_path = os.path.join(dir, file)
        print(f"Processing file: {file_path}")
        loader = PyPDFLoader(file_path)

        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True
        )

        embed_model = None

        if using_zhipu:
            MODEL_API_KEY = config["ZHIPU_API_KEY"]

            embed_model = ZhipuAIEmbeddings(
                model="embedding-3",
                api_key=MODEL_API_KEY,
                dimensions=2048
            )
        else:
            # use cohere model
            MODEL_API_KEY = config["COHERE_API_KEY"]
            print(MODEL_API_KEY)
            embed_model = CohereEmbeddings(
                model="embed-english-v3.0",
                cohere_api_key=MODEL_API_KEY,
            )

        print("set up embedding model")

        docs = text_splitter.split_documents(data)

        print(f"split documents")

        try:
            
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=docs,
                embedding=embed_model,
                collection=collection,
                index_name="vector_index"
            )

            print("created vector store successfully")

        except Exception as e:
            print(f"Error: {e}")
            os._exit(1)

if __name__ == "__main__":
    # get the arguments passed in 
    import sys
    print(sys.argv)
    if len(sys.argv) > 1:
        using_zhipu = False

    main()

    
