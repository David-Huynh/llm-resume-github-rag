""" 
Embeds code files from previous projects using StarEnCoder into a FAISS index
for efficient retrieval in a RAG pipeline.
"""
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
from tqdm import tqdm
import argparse

# Settings
IGNORE_DIRS = {"node_modules", ".git", "venv", "__pycache__", "build", "dist"}
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".java", ".c", ".cpp", ".cc", ".cxx", ".cs", ".go",
    ".rb", ".php", ".swift", ".kt", ".kts", ".scala", ".rs", ".sh", ".bash",
    ".html", ".htm", ".css", ".sql", ".json", ".yaml", ".yml", ".pl", ".pm",
    ".lua", ".r", ".m", ".dart", ".hs", ".jl", ".groovy", ".ps1", ".vb", ".tcl",
    ".f", ".f90", ".adb", ".ads", ".xml", ".md", '.ipynb', '.jsx', '.tsx', ".txt"
}

CHUNK_SIZE = 256
EMBEDDING_DIM = 768
BATCH_SIZE = 16


def read_code_files(root_dir):
    """Read code files from a directory if they are not in ignored 
    directories and are one of the supported file types.

    :param root_dir: The root directory to search for code files.
    :type root_dir: str
    :yield: A tuple containing the file path and its content.
    :rtype: tuple[str, str]
    """
    root_path = Path(root_dir)

    for file_path in root_path.rglob("*"):
        if file_path.is_dir():
            continue  # skip directories
        if any(part in IGNORE_DIRS for part in file_path.parts):
            continue
        if file_path.suffix.lower() in CODE_EXTENSIONS:
            try:
                code = file_path.read_text(encoding="utf-8")
                yield file_path, code
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

def chunk_code(code, chunk_size=CHUNK_SIZE):
    """Chunk code into smaller pieces.

    :param code: The code to chunk.
    :type code: str
    :param chunk_size: The size of each chunk, defaults to CHUNK_SIZE
    :type chunk_size: int, optional
    :yield: A chunk of code.
    :rtype: str
    """
    code = code.strip()
    for i in range(0, len(code), chunk_size):
        yield code[i:i+chunk_size]

def embed_code_chunks(chunks):
    """Embed code chunks using StarEncoder.

    :param chunks: The code chunks to embed.
    :type chunks: list[str]
    :return: The embeddings for the code chunks.
    :rtype: np.ndarray
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("bigcode/starencoder")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained("bigcode/starencoder")
    model.to(device)
    model.eval()
    inputs = tokenizer(list(chunks), padding=True, truncation=True, max_length=1024, return_tensors="pt")
    inputs = {k: v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:,0,:]
    return embeddings.cpu().numpy()

def build_embeddings_index_streaming(root_dir, save_folder):
    """Build embeddings index for code files in a directory.

    :param root_dir: The root directory to search for code files.
    :type root_dir: str
    :param save_folder: The folder to save the FAISS index and metadata.
    :type save_folder: str
    """
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    # Initialize empty FAISS index
    index = faiss.IndexFlatL2(EMBEDDING_DIM)

    metadata_path = save_folder / "metadata.txt"
    metadata_file = open(metadata_path, "w", encoding="utf-8")

    total_chunks = 0

    # Initialize buffers to batch chunks across files
    all_chunks = []
    all_metadata = []

    print("Processing files and building index incrementally...")

    for file_path, code in tqdm(read_code_files(root_dir)):
        chunks = list(chunk_code(code))
        # Append chunks and metadata for this file
        all_chunks.extend(chunks)
        all_metadata.extend([f"{file_path}::{CHUNK_SIZE*i}" for i in range(len(chunks))])

        # When we have enough chunks, embed and add to index
        while len(all_chunks) >= BATCH_SIZE:
            batch_chunks = all_chunks[:BATCH_SIZE]
            batch_embeddings = embed_code_chunks(batch_chunks)
            index.add(batch_embeddings)

            # Write corresponding metadata
            for meta in all_metadata[:BATCH_SIZE]:
                metadata_file.write(meta + "\n")

            total_chunks += BATCH_SIZE
            # Remove processed chunks and metadata from buffers
            all_chunks = all_chunks[BATCH_SIZE:]
            all_metadata = all_metadata[BATCH_SIZE:]

    # Process remaining chunks < BATCH_SIZE after finishing all files
    if all_chunks:
        batch_embeddings = embed_code_chunks(all_chunks)
        index.add(batch_embeddings)
        for meta in all_metadata:
            metadata_file.write(meta + "\n")
        total_chunks += len(all_chunks)

    metadata_file.close()

    print(f"Total chunks indexed: {total_chunks}")

    # Save FAISS index to disk
    faiss.write_index(index, str(save_folder / "faiss.index"))

    print(f"FAISS index and metadata saved to {save_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streaming embedding & indexing with StarEncoder")
    parser.add_argument("--root_dir", help="Directory containing code projects")
    parser.add_argument("--save_folder", help="Folder to save index and metadata")
    args = parser.parse_args()

    build_embeddings_index_streaming(args.root_dir, args.save_folder)
