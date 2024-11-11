import json
from bson import json_util
from time import sleep
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_cohere import CohereEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from dotenv import dotenv_values
import os
import pdb
    
config = dotenv_values(".env")

MONGO_USER = config["MONGO_USER"]
MONGO_PASS = config["MONGO_PASS"]
mongo_connection_string = f"mongodb+srv://{MONGO_USER}:{MONGO_PASS}@ai-cluster.ffx00.mongodb.net/D01?retryWrites=true&w=majority&appName=AI-Cluster"
mongo_client = MongoClient(mongo_connection_string)
collection = mongo_client["D01"]["pdf-test"]

dir = "./pdfs"

using_zhipu = True

filename = ""
query = "What is the purpose of the report?"


print(collection.count_documents({}))

def main():
    print("we're using zhipu:", using_zhipu)
    
    print(f"Processing file: {filename}")
    loader = PyPDFLoader(filename)

    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )

    docs = text_splitter.split_documents(data)
    print(f"split documents")

    # breakpoint()

    embeddings = None

    if using_zhipu:
        MODEL_API_KEY = config["ZHIPU_API_KEY"]

        embeddings = ZhipuAIEmbeddings(
            model="embedding-3",
            api_key=MODEL_API_KEY,
            dimensions=2048
        )
    else:
        # use cohere model
        MODEL_API_KEY = config["COHERE_API_KEY"]
        embeddings = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=MODEL_API_KEY,
        )

    print("set up embedding model")


    def create_and_insert_document(doc):
        # Create document with required structure
        document = {
            "text": doc.page_content,  # This is the required field name
            "embedding": embeddings.embed_query(doc.page_content),
            "metadata": doc.metadata
        }
        
        # Insert into MongoDB
        collection.insert_one(document)
        return document


    try:
        collection.delete_many({})
        print("All documents deleted")

        vector_store = MongoDBAtlasVectorSearch(
            collection, embeddings, 
            index_name="test_index",
            text_key="text",  # Explicitly specify the text field
        )

        for doc in docs:
            create_and_insert_document(doc)

        print("created vector store successfully")

        sleep(4)

        doc_count = collection.count_documents({})
        print(f"\nTotal documents in collection: {doc_count}")

        print("\nChecking for vector index:")
        vector_index = collection.list_indexes()
        for index in vector_index:
            print(index)

        # Function to perform search and display results
        def test_similarity_search(query, k=3):
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            results = vector_store.similarity_search(query, k=k)
            
            for i, doc in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Content: {doc.page_content[:200]}...")
                print(f"Source: Page {doc.metadata.get('page', 'N/A')}")
                print("-" * 30)


        test_similarity_search(query)

    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)

if __name__ == "__main__":
    # get the arguments passed in 
    import sys
    print(sys.argv)
    if len(sys.argv) < 3:
        print('run this as: python individual_test.py filename.pdf "query"')
        os._exit(1)

    filename = sys.argv[1]
    query = sys.argv[2]
    print(f"filename: {filename}, query: {query}")

    main()

    
