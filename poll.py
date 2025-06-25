import os
import json
import hashlib
import logging
from azure.storage.blob import BlobServiceClient, ContainerClient
from manual_retrieval.chunking import chunk_markdown_file
from manual_retrieval.tree_utils import generate_leaf_nodes, build_balanced_tree
from dotenv import load_dotenv

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONTAINER = "ds100-su25"
PREFIX_RAW = "docs_manual/raw/"
PREFIX_CHUNKS = "docs_manual/chunks/"
PREFIX_TREES = "docs_manual/trees/"
HASHES_BLOB_PATH = f"{PREFIX_RAW}hashes.json"

load_dotenv("keys.env")
blob_service = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service.get_container_client(CONTAINER)

def compute_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

def load_hashes():
    try:
        blob_client = container_client.get_blob_client(HASHES_BLOB_PATH)
        return json.loads(blob_client.download_blob().readall().decode("utf-8"))
    except:
        return {}

def save_hashes(hashes: dict):
    blob_client = container_client.get_blob_client(HASHES_BLOB_PATH)
    blob_client.upload_blob(json.dumps(hashes, indent=2), overwrite=True)

def create_TOC(tree_prefix: str, container_client: ContainerClient):
    table_of_contents = {}

    blobs = container_client.list_blobs(name_starts_with=tree_prefix)
    for blob in sorted(blobs, key=lambda b: b.name):
        filename = os.path.basename(blob.name)
        if not filename.endswith(".json") or "table_of_contents" in filename:
            continue

        blob_client = container_client.get_blob_client(blob.name)
        content = blob_client.download_blob().readall().decode("utf-8")
        data = json.loads(content)

        highest_key = max(data.keys())
        table_of_contents[filename] = highest_key

    toc_blob_path = f"{tree_prefix}table_of_contents.json"
    toc_blob = container_client.get_blob_client(toc_blob_path)
    toc_blob.upload_blob(json.dumps(table_of_contents, indent=4), overwrite=True)

    print("table_of_contents.json created successfully in Azure!")

def upload_json(blob_path, data):
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(json.dumps(data, indent=2), overwrite=True)

def build_tree(chunks, branch_factor=3):
    leaf_nodes = generate_leaf_nodes(chunks)
    return build_balanced_tree(leaf_nodes, branch_factor)

def run():
    blobs = list(container_client.list_blobs(name_starts_with=PREFIX_RAW))
    raw_md_blobs = [b for b in blobs if b.name.endswith(".md")]
    current_filenames = {os.path.basename(b.name) for b in raw_md_blobs}

    hashes = load_hashes()
    updated_hashes = dict(hashes)
    processed = False

    # Step 1: Process new or changed .md files
    for blob in raw_md_blobs:
        filename = os.path.basename(blob.name)
        md_blob = container_client.get_blob_client(blob.name)
        md_content = md_blob.download_blob().readall().decode("utf-8")
        new_hash = compute_hash(md_content)

        if filename in hashes and hashes[filename] == new_hash:
            continue  # unchanged

        logger.info(f"Processing updated or new file: {filename}")
        updated_hashes[filename] = new_hash

        chunks = chunk_markdown_file(md_content)
        json_filename = filename.replace(".md", ".json")

        # Upload chunks
        upload_json(f"{PREFIX_CHUNKS}{json_filename}", chunks)

        # Build and upload tree
        tree = build_tree(chunks)
        upload_json(f"{PREFIX_TREES}{json_filename}", tree)

        processed = True

    # Step 2: Remove hash entries for files no longer present
    removed = set(hashes.keys()) - current_filenames
    for fname in removed:
        logger.info(f"Removing stale hash entry for {fname}")
        updated_hashes.pop(fname, None)

    save_hashes(updated_hashes)

    # Step 3: Handle chunked files missing trees
    chunk_blobs = container_client.list_blobs(name_starts_with=PREFIX_CHUNKS)
    tree_blobs_set = {
        os.path.basename(blob.name)
        for blob in container_client.list_blobs(name_starts_with=PREFIX_TREES)
    }

    for chunk_blob in chunk_blobs:
        json_filename = os.path.basename(chunk_blob.name)
        if json_filename not in tree_blobs_set:
            logger.info(f"Building missing tree for {json_filename}...")
            content = container_client.get_blob_client(chunk_blob.name).download_blob().readall().decode("utf-8")
            chunks = json.loads(content)
            tree = build_tree(chunks)
            upload_json(f"{PREFIX_TREES}{json_filename}", tree)
            processed = True

    if processed:
        logger.info("Updating TOC...")
        create_TOC(PREFIX_TREES, container_client)
        logger.info("TOC updated!")

if __name__ == "__main__":
    run()
