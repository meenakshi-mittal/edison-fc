import os
import json
from azure.storage.blob import BlobServiceClient, ContainerClient
from manual_retrieval.chunking import chunk_markdown_file
from manual_retrieval.tree_utils import generate_leaf_nodes, build_balanced_tree
from dotenv import load_dotenv
import logging


logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CONTAINER = "ds100-su25"
PREFIX_RAW = "docs_manual/raw/"
PREFIX_CHUNKS = "docs_manual/chunks/"
PREFIX_TREES = "docs_manual/trees/"

load_dotenv("keys.env")
blob_service = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
container_client = blob_service.get_container_client(CONTAINER)

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

def already_chunked(md_filename):
    json_filename = md_filename.replace(".md", ".json")
    chunk_blob_path = f"{PREFIX_CHUNKS}{json_filename}"
    return container_client.get_blob_client(chunk_blob_path).exists()

def upload_json(blob_path, data):
    blob_client = container_client.get_blob_client(blob_path)
    blob_client.upload_blob(json.dumps(data, indent=2), overwrite=True)

def build_tree(chunks, branch_factor=3):
    leaf_nodes = generate_leaf_nodes(chunks)
    return build_balanced_tree(leaf_nodes, branch_factor)

def run():
    blobs = container_client.list_blobs(name_starts_with=PREFIX_RAW)
    processed = False

    # Step 1: Process new .md files
    for blob in blobs:
        if not blob.name.endswith(".md"):
            continue
        filename = os.path.basename(blob.name)
        if already_chunked(filename):
            continue

        logger.info(f"Chunking and building tree for {filename}...")
        md_blob = container_client.get_blob_client(blob.name)
        md_content = md_blob.download_blob().readall().decode("utf-8")
        chunks = chunk_markdown_file(md_content)

        json_filename = filename.replace(".md", ".json")

        # Upload chunks
        upload_json(f"{PREFIX_CHUNKS}{json_filename}", chunks)

        # Build and upload tree
        tree = build_tree(chunks)
        upload_json(f"{PREFIX_TREES}{json_filename}", tree)

        processed = True

    # Step 2: Process chunked files missing trees
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

    # if processed:
    logger.info("Updating TOC...")
    create_TOC(PREFIX_TREES, container_client)
    logger.info("TOC updated!")

if __name__ == "__main__":
    run()