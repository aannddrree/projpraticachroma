from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = PersistentClient(path="chroma_db")

collection = client.get_collection(
    name="politicas_loja",
    embedding_function=embedding_fn
)

dados = collection.get(
    include=["embeddings"]
)

print(dados["embeddings"])