import os
import json
from azure.storage.blob import BlobServiceClient
from manual_retrieval.chunking import chunk_markdown_string  # adjust import

CONTAINER = "edisondata"
PREFIX_RAW = "ds100-su25/docs_manual/raw/"
PREFIX_CHUNKS = "ds100-su25/docs_manual/chunks/"

blob_service = BlobServiceClient.from_connection_string(os.environ["AZURE_STORAGE_CONNECTION_STRING"])
container_client = blob_service.get_container_client(CONTAINER)

def already_chunked(md_filename):
    json_filename = md_filename.replace(".md", ".json")
    chunk_blob_path = f"{PREFIX_CHUNKS}{json_filename}"
    return container_client.get_blob_client(chunk_blob_path).exists()

def run():
    blobs = container_client.list_blobs(name_starts_with=PREFIX_RAW)
    for blob in blobs:
        if not blob.name.endswith(".md"):
            continue
        filename = os.path.basename(blob.name)
        if already_chunked(filename):
            continue

        print(f"Chunking {filename}...")
        md_blob = container_client.get_blob_client(blob.name)
        md_content = md_blob.download_blob().readall().decode("utf-8")
        chunks = chunk_markdown_string(md_content)

        json_filename = filename.replace(".md", ".json")
        chunk_blob = container_client.get_blob_client(f"{PREFIX_CHUNKS}{json_filename}")
        chunk_blob.upload_blob(json.dumps(chunks, indent=2), overwrite=True)
