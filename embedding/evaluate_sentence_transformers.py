
"""
Script to evaluate the retrieval performance of the Sentence Transformers model.
"""
import json
import time

import datasets
from langchain_community.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from tqdm import tqdm

if __name__ == "__main__":
    # Load the dataset
    dataset_path = "/path/to/dataset"
    ds = datasets.load_dataset(
        dataset_path,
        split="test",
        cache_dir="/path/to/cache_dir",
    )

    print("Getting documents...")
    # Create LangchainDocument objects from the dataset
    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(page_content=doc["positive"], metadata={
                        "doi": doc["doi"], "_id": doc["_id"]})
        for doc in tqdm(ds)
    ]

    print("Getting queries...")
    # Create query objects from the dataset
    QUERY = [{"query": doc["anchor"], "_id": doc["_id"]} for doc in ds]

    EMBEDDING_MODEL_NAME = 'oe-sroberta-raw-mean'
    EMBEDDING_MODEL_REPO = 'Dingyun-Huang/' + EMBEDDING_MODEL_NAME

    # Initialize the HuggingFaceEmbeddings model
    print("Initializing the HuggingFaceEmbeddings model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_REPO,
        multi_process=True,
        model_kwargs={
            "device": "cuda", 
            "model_kwargs": {"add_pooling_layer": False}
        },
        # Set True for cosine similarity
        encode_kwargs={"normalize_embeddings": True, "batch_size": 1024},
        cache_folder="/path/to/cache_dir",
        show_progress=True,
    )

    
    # Create the knowledge vector database using FAISS
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        RAW_KNOWLEDGE_BASE, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )  # comment out this line to load the knowledge vector database from local

    # Save the knowledge vector database
    KNOWLEDGE_VECTOR_DATABASE.save_local(
        "/path/to/local/index",
        f"{EMBEDDING_MODEL_NAME}-index"
    ) # comment out this line to load the knowledge vector database from local

    """
    
    # Load the knowledge vector database
    KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
        "/grand/SolarWindowsADSP/dingyun/optoelectronics/retrieval/indexes/",
        embedding_model,
        f"{EMBEDDING_MODEL_NAME}-index",
        allow_dangerous_deserialization=True,
    )
    
    """

    # batch embed queries
    print("Embedding queries...")
    time1 = time.time()
    QUERY_EMBEDDINGS = embedding_model.embed_documents(
        [query["query"] for query in QUERY]
    )
    time2 = time.time()
    print(f"Embedding time: {time2 - time1}")

    for i, e in enumerate(QUERY_EMBEDDINGS):
        QUERY[i]["embedding"] = e
    
    # Retrieve the relevant documents for the given query
    retrieved_ids = []
    k_max = 20
    for q in tqdm(QUERY):
        # print(f"Query: {q["_id"]}")
        relevant_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search_by_vector(
            embedding=q["embedding"], k=k_max
        )
        relevant_ids = [doc.metadata["_id"] for doc in relevant_docs]
        retrieved_ids.append(relevant_ids)
        # break

    def generate_recalls(k):
        for i, ids in enumerate(retrieved_ids):
            if QUERY[i]["_id"] in ids[:k]:
                yield 1

    for k in range(1, k_max + 1):
        recall = sum(generate_recalls(k))
        print(f"Recall@{k}: {recall / len(QUERY)}")

    with open(f"retrieved-ids-{EMBEDDING_MODEL_NAME}.json", "w", encoding="utf8") as f:
        for i, ids in enumerate(retrieved_ids):
            f.write(json.dumps({"query_id": QUERY[i]["_id"], "retrieved_ids": ids}) + "\n")
