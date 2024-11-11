import json
from bson import json_util
from time import sleep
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

using_zhipu = False
filename = ""

def main():
    print("we're using zhipu:", using_zhipu)
    
    print(f"Processing file: {filename}")
    loader = PyPDFLoader(filename)

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

        sleep(4)

        from pymongo.operations import SearchIndexModel

        try:
            # Create the search index model
            index_model = SearchIndexModel(
                {
                    "mappings": {
                        "dynamic": True,
                        "fields": {
                            "embedding": {
                                "type": "knnVector",
                                "dimensions": 1024,
                                "similarity": "cosine"
                            }
                        }
                    }
                },
                name="vector_index"
            )
            
            # Create the index
            collection.create_search_index(index_model)
            print("Vector index created successfully!")
            
            # Verify the index was created
            print("\nCurrent indexes:")
            indexes = collection.list_indexes()
            for index in indexes:
                print(index)

        except Exception as e:
            print(f"Error creating index: {str(e)}")

        # Count total documents
        doc_count = collection.count_documents({})
        print(f"\nTotal documents in collection: {doc_count}")

        # Let's also specifically check for the vector index
        print("\nChecking for vector index:")
        vector_index = collection.list_indexes()
        for index in vector_index:
            if "vector_index" in str(index):
                print(f"Found vector index: {index}")

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

        # Example queries to test
        test_queries = [
            "What is the relationship between government repression and opposition group violence?",
            "How do opposition groups choose between violent and nonviolent tactics?",
            "What are the main findings about deterrence effectiveness?",
            "Explain the Rational Actor model versus Action-Reaction model",
            "What factors influence whether repression increases or decreases dissent?"
        ]

        # Run tests
        for query in test_queries:
            test_similarity_search(query)

    except Exception as e:
        print(f"Error: {e}")
        os._exit(1)

if __name__ == "__main__":
    # get the arguments passed in 
    import sys
    print(sys.argv)
    if len(sys.argv) < 2:
        print("run this as: python individual_test.py filename")
        os._exit(1)

    filename = sys.argv[1]

    main()

    
