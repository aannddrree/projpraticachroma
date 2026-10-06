"""Aula 6 — exemplo executável de banco vetorial com Chroma.

Instalação:
    pip install chromadb sentence-transformers
"""

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


DOCUMENTOS = [
    (
        "trocas-01",
        "Você pode devolver calçados em até 30 dias, desde que não tenham sido usados.",
        {"categoria": "trocas", "produto": "calcado"},
    ),
    (
        "frete-01",
        "Entregamos em todo o Brasil. O prazo varia conforme o CEP informado.",
        {"categoria": "entrega", "produto": "geral"},
    ),
    (
        "pagamento-01",
        "Aceitamos cartão de crédito, PIX e boleto bancário.",
        {"categoria": "pagamento", "produto": "geral"},
    ),
    (
        "trocas-02",
        "Para trocar uma roupa, mantenha as etiquetas e solicite a troca pelo atendimento.",
        {"categoria": "trocas", "produto": "roupa"},
    ),
    (
        "garantia-01",
        "Produtos com defeito de fabricação têm garantia de 90 dias.",
        {"categoria": "garantia", "produto": "geral"},
    ),
    (
        "cadastro-01",
        "Para acompanhar pedidos, entre na área Minha Conta com seu e-mail cadastrado.",
        {"categoria": "conta", "produto": "geral"},
    ),
]


def main() -> None:
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="politicas_loja",
        embedding_function=embedding_fn,
    )

    collection.upsert(
        ids=[documento[0] for documento in DOCUMENTOS],
        documents=[documento[1] for documento in DOCUMENTOS],
        metadatas=[documento[2] for documento in DOCUMENTOS],
    )
    print(f"Documentos indexados: {collection.count()}")

    pergunta = "Como devolvo um calçado que comprei?"
    resultado = collection.query(
        query_texts=[pergunta],
        n_results=3,
        include=["documents", "metadatas", "distances"],
    )

    print(f"\nPergunta: {pergunta}\n")
    for texto, metadata, distancia in zip(
        resultado["documents"][0],
        resultado["metadatas"][0],
        resultado["distances"][0],
    ):
        print(f"Distância: {distancia:.4f}")
        print(f"Categoria: {metadata['categoria']}")
        print(f"Trecho: {texto}\n")

    somente_trocas = collection.query(
        query_texts=[pergunta],
        n_results=2,
        where={"categoria": "trocas"},
        include=["documents"],
    )
    print("Busca filtrada na categoria 'trocas':")
    for texto in somente_trocas["documents"][0]:
        print(f"- {texto}")


if __name__ == "__main__":
    main()
