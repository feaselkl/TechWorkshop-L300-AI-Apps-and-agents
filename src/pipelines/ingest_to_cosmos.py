import logging
import json
import os
from typing import Any

from azure.cosmos import CosmosClient, PartitionKey
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import AzureError
from dotenv import load_dotenv

load_dotenv()

# CONFIGURATIONS - Replace with your actual values or set as env vars
COSMOS_ENDPOINT = os.environ.get("COSMOS_ENDPOINT")
COSMOS_KEY = os.environ.get("COSMOS_KEY")
DATABASE_NAME = os.environ.get("DATABASE_NAME")
CONTAINER_NAME = os.environ.get("CONTAINER_NAME")
JSON_FILE = os.environ.get("JSON_FILE", "data/product_catalog.json")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def get_cosmos_client(endpoint: str | None, key: str | None = None):
    if not endpoint:
        raise ValueError("COSMOS_ENDPOINT must be provided in environment variables")

    # Try Entra ID first
    try:
        logger.info("Attempting to authenticate to Cosmos DB using DefaultAzureCredential (managed identity)...")
        credential = DefaultAzureCredential()
        client = CosmosClient(endpoint, credential=credential)
        _ = list(client.list_databases())
        logger.info("Authenticated to Cosmos DB with DefaultAzureCredential.")
        return client
    except AzureError as ex:
        logger.warning("Managed identity authentication failed: %s", ex)

    # Fallback to key
    if key:
        try:
            logger.info("Falling back to endpoint + key authentication for Cosmos DB...")
            client = CosmosClient(endpoint, key)
            _ = list(client.list_databases())
            logger.info("Authenticated to Cosmos DB with endpoint+key.")
            return client
        except Exception as ex:
            logger.error("Endpoint+key authentication failed: %s", ex)
            raise

    raise RuntimeError(
        "Failed to authenticate to Cosmos DB using DefaultAzureCredential and no valid COSMOS_KEY was provided"
    )


def load_json_items(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported JSON structure in file")


def ensure_string_ids(item: dict[str, Any]) -> dict[str, Any]:
    # Ensure ProductID exists and both id and ProductID are strings
    if "ProductID" not in item:
        raise KeyError("Item missing 'ProductID' field")

    product_id = item["ProductID"]
    # Keep original value but convert to string for id and ProductID
    item["ProductID"] = str(product_id)
    item["id"] = str(product_id)

    # Build content_for_vector similar to original script
    name = str(item.get("ProductName", ""))
    category = str(item.get("ProductCategory", ""))
    desc = str(item.get("ProductDescription", ""))

    return item


def main() -> None:
    client = get_cosmos_client(COSMOS_ENDPOINT, COSMOS_KEY)

    if not DATABASE_NAME:
        raise ValueError("DATABASE_NAME must be provided in environment variables")

    if not CONTAINER_NAME:
        raise ValueError("CONTAINER_NAME must be provided in environment variables")

    database = client.create_database_if_not_exists(id=DATABASE_NAME)
    container = database.create_container_if_not_exists(
        id=CONTAINER_NAME, partition_key=PartitionKey(path="/ProductID")
    )

    items = load_json_items(JSON_FILE)

    for raw in items:
        try:
            item = ensure_string_ids(dict(raw))
            container.upsert_item(body=item)
            print(f"Uploaded: ProductID {item['ProductID']}")
        except Exception as ex:
            logger.error("Failed to upload item: %s; error: %s", raw, ex)

    print("All data uploaded to Cosmos DB.")


if __name__ == "__main__":
    main()