# importing libraries
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore 
from langchain_core.documents import Document
from typing import List
import chromadb
import json
import re

# initializing the text splitter
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)

# initializing the embedder
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# defining the db
vectorstore = Chroma(
    persist_directory="/workspace/topic_identifier/chroma_db",
    embedding_function=embedding_model
)
store = InMemoryStore()

# defining the retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store, 
    parent_splitter=parent_splitter,
    child_splitter=child_splitter   
)

def get_topics(vectorstore):
    """Get all unique topics from the vectorstore"""
    try:
        client = chromadb.PersistentClient(path="/workspace/topic_identifier/chroma_db")
        collection_name = "test_documents"
        collection = client.get_collection(collection_name)
        results = collection.get(include=['metadatas'])
        dict_topics = results['metadatas']
        topics = list(set(dict['topic'] for dict in dict_topics if 'topic' in dict))
        print(f"Found topics: {topics}")
        return topics
    except Exception as e:
        print(f"Error getting topics: {e}")
        return []

def get_parent_docs(topic):
    """Get documents for a specific topic - multiple approaches"""
    print(f"Searching for documents with topic: '{topic}'")
    
    # Method 1: Try direct ChromaDB query first
    try:
        client = chromadb.PersistentClient(path="/workspace/topic_identifier/chroma_db")
        collection = client.get_collection("test_documents")
        
        # Query with where clause
        results = collection.get(
            where={"topic": topic},
            include=['documents', 'metadatas']
        )
        
        if results['documents']:
            print(f"Found {len(results['documents'])} documents using direct ChromaDB query")
            docs = []
            for i, doc_content in enumerate(results['documents']):
                metadata = results['metadatas'][i] if i < len(results['metadatas']) else {}
                docs.append(Document(page_content=doc_content, metadata=metadata))
            return docs
    except Exception as e:
        print(f"ChromaDB direct query failed: {e}")
    
    # Method 2: Try Langchain vectorstore similarity search with filter
    try:
        docs = vectorstore.similarity_search(
            query=f"information about {topic}",  # Use actual query instead of empty string
            k=100,
            filter={"topic": topic}
        )
        if docs:
            print(f"Found {len(docs)} documents using vectorstore similarity search with filter")
            return docs
    except Exception as e:
        print(f"Vectorstore similarity search with filter failed: {e}")
    
    # Method 3: Try similarity search without filter and manually filter
    try:
        all_docs = vectorstore.similarity_search(
            query=f"information about {topic}",
            k=1000  # Get more docs to filter from
        )
        filtered_docs = [doc for doc in all_docs if doc.metadata.get('topic') == topic]
        if filtered_docs:
            print(f"Found {len(filtered_docs)} documents using manual filtering")
            return filtered_docs
    except Exception as e:
        print(f"Manual filtering approach failed: {e}")
    
    # Method 4: Try using the retriever directly
    try:
        retrieved_docs = retriever.get_relevant_documents(f"information about {topic}")
        topic_docs = [doc for doc in retrieved_docs if doc.metadata.get('topic') == topic]
        if topic_docs:
            print(f"Found {len(topic_docs)} documents using retriever")
            return topic_docs
    except Exception as e:
        print(f"Retriever approach failed: {e}")
    
    print(f"No documents found for topic: {topic}")
    return []

def extract_json_array(output):
    """Extract JSON array from output"""
    match = re.search(r'\[.*?\]', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            print("Matched text is not valid JSON")
            return None
    else:
        print("No JSON array found")
        return None

def generate_mcqs(topic: str, chunk: str) -> str:
    """Generate MCQs for a given topic and chunk"""
    try:
        prompt_template = PromptTemplate(
            input_variables=["topic", "chunk"],
            template="""
You are a test-set generator. Based on the following content about "{topic}", generate 2 multiple choice questions (each with 4 options and 1 correct answer):

Content:
{chunk}

Format each question as:
Q: Question?
a) Option A
b) Option B
c) Option C
d) Option D
Answer: <correct letter>

Please provide exactly 2 questions in this format.
            """.strip()
        )
        
        model = OllamaLLM(model="deepseek-r1:1.5b")
        chain = prompt_template | model | StrOutputParser()
        
        # Truncate chunk if too long to avoid token limits
        max_chunk_size = 2000
        if len(chunk) > max_chunk_size:
            chunk = chunk[:max_chunk_size] + "..."
        
        full_question = chain.invoke({"topic": topic, "chunk": chunk})
        return full_question
    except Exception as e:
        print(f"Error generating MCQs for topic {topic}: {e}")
        return None

# Debug: Check vectorstore contents
def debug_vectorstore():
    """Debug function to check what's in the vectorstore"""
    try:
        client = chromadb.PersistentClient(path="/workspace/topic_identifier/chroma_db")
        collections = client.list_collections()
        print(f"Available collections: {[c.name for c in collections]}")
        
        if collections:
            collection = collections[0]  # Get first collection
            results = collection.get(include=['metadatas'])
            print(f"Sample metadata: {results['metadatas'][:3] if results['metadatas'] else 'No metadata'}")
            
            # Check for topic field variations
            if results['metadatas']:
                all_keys = set()
                for metadata in results['metadatas']:
                    all_keys.update(metadata.keys())
                print(f"All metadata keys found: {all_keys}")
    except Exception as e:
        print(f"Debug failed: {e}")

# Run debug first
print("=== DEBUG INFO ===")
debug_vectorstore()
print("=== END DEBUG ===\n")

# Get all topics
topics = get_topics(vectorstore)
print(f"Processing {len(topics)} topics: {topics}\n")

all_generated_questions = {}

for topic in topics:
    print(f"Processing topic: {topic}")
    parent_docs = get_parent_docs(topic)
    
    if not parent_docs:
        print(f"No documents found for topic: {topic}\n")
        continue
    
    print(f"Found {len(parent_docs)} documents for topic: {topic}")
    test_questions = []
    
    for i, doc in enumerate(parent_docs):
        print(f"Generating MCQ for document {i+1}/{len(parent_docs)}")
        quiz = generate_mcqs(topic, doc.page_content)
        if quiz:
            test_questions.append(quiz)
        else:
            print(f"Failed to generate question for document {i+1}")
    
    all_generated_questions[topic] = test_questions
    print(f"Generated {len(test_questions)} questions for topic: {topic}\n")

# Print final result
print("=== FINAL RESULTS ===")
print(json.dumps(all_generated_questions, indent=2))