import json
import chromadb
from chromadb.config import Settings
from openai import OpenAI

client = OpenAI()

chroma = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="db"))

def load_chunks(path):
    chunks = []
    with open(path, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def embed_text(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def build_index(name, chunks):
    collection = chroma.get_or_create_collection(name=name)

    ids = []
    docs = []
    metas = []
    embeds = []

    for c in chunks:
        ids.append(c["chunk_id"])
        docs.append(c["text"])
        metas.append(c["metadata"])
        embeds.append(embed_text(c["text"]))

    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeds
    )

    print(f"Indexed {len(ids)} chunks into {name}")

def main():
    semantic_chunks = load_chunks("semantic_chunks.jsonl")
    recursive_chunks = load_chunks("recursive_chunks.jsonl")

    build_index("semantic_index", semantic_chunks)
    build_index("recursive_index", recursive_chunks)

if __name__ == "__main__":
    main()
