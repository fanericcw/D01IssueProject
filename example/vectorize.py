# https://python.langchain.com/docs/modules/data_connection/vectorstores/integrations/mongodb_atlas

from langchain_cohere import CohereEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import params

# Step 1: Load
# loaders = [
#  WebBaseLoader("https://en.wikipedia.org/wiki/AT%26T"),
#  WebBaseLoader("https://en.wikipedia.org/wiki/Bank_of_America")
# ]
# data = []
# for loader in loaders:
#     data.extend(loader.load())

PDF_FILE = "../report-2-8.pdf"
loader = PyPDFLoader(PDF_FILE)
data = loader.load()

# Step 2: Transform (Split)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separators=[
                                               "\n\n", "\n", "(?<=\. )", " "], length_function=len)
docs = text_splitter.split_documents(data)
print('Split into ' + str(len(docs)) + ' docs')

# Step 3: Embed
# https://api.python.langchain.com/en/latest/embeddings/langchain.embeddings.openai.OpenAIEmbeddings.html
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=params.cohere_api_key,
)

# Step 4: Store
# Initialize MongoDB python client
client = MongoClient(params.mongodb_conn_string)
collection = client[params.db_name][params.collection_name]

# Reset w/out deleting the Search Index 
collection.delete_many({})
print("All documents deleted")

# # Drop all indexes except the default _id index
# collection.drop_indexes()
# print("All indexes dropped")

print(collection.count_documents({}))
# Insert the documents in MongoDB Atlas with their embedding
# https://github.com/hwchase17/langchain/blob/master/langchain/vectorstores/mongodb_atlas.py
# docsearch = MongoDBAtlasVectorSearch.from_documents(
#     docs, embeddings, collection=collection, index_name=params.index_name
# )

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

for doc in docs:
    create_and_insert_document(doc)

print(collection.count_documents({}))
