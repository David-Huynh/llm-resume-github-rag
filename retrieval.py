""" 
Functions for retrieving code snippets from a FAISS index given
a query string. ie. A job posting.
"""
import faiss
from transformers import AutoTokenizer, AutoModel
import torch

def embed_query(query):
    """Embed a query string into a vector representation.

    :param query: The query string to embed.
    :type query: str
    :return: The embedded vector representation of the query.
    :rtype: np.ndarray
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bigcode/starencoder")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained("bigcode/starencoder").to(device)
    model.eval()
    inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:,0,:].cpu().numpy()
    return emb.astype("float32")

def search_code(query, index_path, metadata_path, k=5):
    """Search for relevant code snippets in the index.

    :param query: The query string to search for.
    :type query: str
    :param index_path: The path to the FAISS index file.
    :type index_path: str
    :param metadata_path: The path to the metadata file.
    :type metadata_path: str
    :param k: The number of top results to return, defaults to 5
    :type k: int, optional
    :return: A list of tuples containing the file path, offset, and distance for each result.
    :rtype: list[tuple[str, int, float]]
    """
    query_emb = embed_query(query)
    # Load index
    index = faiss.read_index(index_path)
    # Load metadata
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = f.read().splitlines()

    D, I = index.search(query_emb, k)  # D = distances, I = indices
    results = []
    for idx, dist in zip(I[0], D[0]):
        file_info = metadata[idx]  # e.g. "/path/to/file.py::512"
        file_path, offset_str = file_info.split("::")
        offset = int(offset_str)
        results.append((file_path, offset, dist))
    return results

def get_code_chunk(file_path, offset, chunk_size=256):
    """Retrieve a chunk of code from a file.

    :param file_path: The path to the code file.
    :type file_path: str
    :param offset: The offset position to start the chunk.
    :type offset: int
    :param chunk_size: The size of the chunk to retrieve, defaults to 256
    :type chunk_size: int, optional
    :return: The retrieved code chunk.
    :rtype: str
    """
    with open(file_path, "r", encoding="utf-8") as f:
        f.seek(0)
        code = f.read()
    return code[offset:offset+chunk_size]


def retrieve_chunks_for_query(query, index_path, metadata_path, k=5, chunk_size=256):
    """Retrieve relevant code chunks for a given query.

    :param query: The query string to search for.
    :type query: str
    :param index_path: The path to the FAISS index file.
    :type index_path: str
    :param metadata_path: The path to the metadata file.
    :type metadata_path: str
    :param k: The number of top results to return, defaults to 5
    :type k: int, optional
    :param chunk_size: The size of the code chunks to retrieve, defaults to 256
    :type chunk_size: int, optional
    :return: A list of code chunks relevant to the query.
    :rtype: list[dict]
    """

    results = search_code(query, index_path, metadata_path, k)
    chunks = []
    for file_path, offset, dist in results:
        chunk_text = get_code_chunk(file_path, offset, chunk_size)
        chunks.append({
            "file_path": file_path,
            "offset": offset,
            "distance": dist,
            "text": chunk_text
        })
    return chunks
