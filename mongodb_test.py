from time import sleep
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import json
from bson import json_util
from dotenv import dotenv_values

config = dotenv_values(".env")

MONGO_USER = config["MONGO_USER"]
MONGO_PASS = config["MONGO_PASS"]
mongo_connection_string = f"mongodb+srv://{MONGO_USER}:{MONGO_PASS}@ai-cluster.ffx00.mongodb.net/D01?retryWrites=true&w=majority&appName=AI-Cluster"
client = MongoClient(mongo_connection_string)
collection = client["D01"]["pdf-test"]

COHERE_API_KEY = config["COHERE_API_KEY"]

PDF_FILE = "./report-2-8.pdf"

def reset_database():
    """Clear everything from the collection and drop indexes"""
    try:
        # Drop all documents
        collection.delete_many({})
        print("All documents deleted")
        
        # Drop all indexes except the default _id index
        collection.drop_indexes()
        print("All indexes dropped")
        
        # Verify
        doc_count = collection.count_documents({})
        print(f"Document count after reset: {doc_count}")
    except Exception as e:
        print(f"Error during reset: {str(e)}")

def create_vector_index():
    """Create the vector search index"""
    try:
        indexes = collection.list_indexes()
        if any(index.get("name") == "vector_index" for index in indexes):
            collection.drop_index("vector_index")
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
        collection.create_search_index(index_model)
        print("Vector index created successfully!")
    except Exception as e:
        print(f"Error creating index: {str(e)}")

def process_pdf():
    """Load, split and embed PDF content"""
    try:
        # Initialize embeddings
        embed_model = CohereEmbeddings(
            model="embed-english-v3.0",
            cohere_api_key=COHERE_API_KEY
        )
        
        # Load PDF
        loader = PyPDFLoader(PDF_FILE)
        data = loader.load()
        print("PDF loaded successfully")
        
        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=len,
            add_start_index=True
        )
        docs = text_splitter.split_documents(data)
        print(f"Split into {len(docs)} documents")
        
        # Create vector store
        vector_store = MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embed_model,
            collection=collection,
            index_name="vector_index"
        )
        print("Vector store created")
        
        return vector_store
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

def verify_setup():
    """Verify the database setup"""
    try:
        # Check document count
        doc_count = collection.count_documents({})
        print(f"\nTotal documents: {doc_count}")
        
        # Check indexes
        print("\nCurrent indexes:")
        indexes = collection.list_indexes()
        for index in indexes:
            print(index)
        
    except Exception as e:
        print(f"Error during verification: {str(e)}")

def test_search(vector_store):
    """Test the similarity search"""
    try:
        # Test queries
        test_queries = [
            "What brought about the rise of digital labour platforms?",
            "What are the effects of the rise of the platform economy?",
            "How do platforms affect low-income workers?"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            print("-" * 50)
            
            results = vector_store.similarity_search(query, k=2)
            
            if results:
                for i, doc in enumerate(results, 1):
                    print(f"\nResult {i}:")
                    print(f"Content: {doc.page_content[:200]}...")
                    print("-" * 30)
            else:
                print("No results found")
                
    except Exception as e:
        print(f"Error during search: {str(e)}")

def main():
    # 1. Reset everything
    print("Resetting database...")
    reset_database()

    sleep(5)
    
    # 2. Create vector index
    print("\nCreating vector index...")
    create_vector_index()
    
    # 3. Process PDF and create vector store
    print("\nProcessing PDF...")
    vector_store = process_pdf()
    
    # 4. Verify setup
    print("\nVerifying setup...")
    verify_setup()
    
    # 5. Test search if vector store was created successfully
    if vector_store:
        print("\nTesting search...")
        test_search(vector_store)
    
    client.close()

if __name__ == "__main__":
    main()