import sys
import logging
import os

# Ensure project root is on PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.config.settings import settings

def main():
    try:
        from pinecone import Pinecone, ServerlessSpec
    except Exception as e:
        logging.error(f"Pinecone SDK not installed: {e}")
        sys.exit(1)

    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index_name = settings.PINECONE_INDEX_NAME

    # List existing indexes
    existing_names = [idx["name"] for idx in pc.list_indexes()]
    if index_name in existing_names:
        desc = pc.describe_index(index_name)
        dim = desc.get('dimension')
        metric = desc.get('metric')
        print(f"Index '{index_name}' already exists. Dimension={dim}, Metric={metric}")
        if dim != 1024 or metric != 'cosine':
            print("Incompatible index configuration! Expected dimension=1024 and metric='cosine'.")
            sys.exit(1)
        else:
            print("Index configuration is compatible.")
        return

    # Create index with ServerlessSpec (default region)
    region = "us-east-1"  # Adjust if your Pinecone project uses a different region
    spec = ServerlessSpec(cloud="aws", region=region)
    pc.create_index(name=index_name, dimension=1024, metric="cosine", spec=spec)
    print(f"Created Pinecone index '{index_name}' with dimension=1024, metric='cosine', region={region}.")

if __name__ == "__main__":
    main()
