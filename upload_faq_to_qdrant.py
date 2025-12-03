#!/usr/bin/env python3
"""
Upload FAQ and Law data to Qdrant Cloud
This script reads data from faq (1).json and law.json and uploads to Qdrant Cloud
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, PayloadSchemaType
import uuid
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Configuration
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if credentials are set
if not QDRANT_CLOUD_URL or not QDRANT_API_KEY:
    print("❌ Error: Qdrant Cloud credentials not found!")
    print("Please set QDRANT_CLOUD_URL and QDRANT_API_KEY in your .env file")
    exit(1)

if not OPENAI_API_KEY:
    print("❌ Error: OpenAI API key not found!")
    print("Please set OPENAI_API_KEY in your .env file")
    exit(1)

print("🔧 Initializing...")
print(f"📍 Qdrant Cloud URL: {QDRANT_CLOUD_URL}")

# Initialize embeddings and Qdrant client
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)

VECTOR_SIZE = 1536  # text-embedding-3-small produces 1536-dimensional vectors


# ========== HELPER FUNCTIONS ==========

def extract_dieu_number(dieu_text: str) -> int:
    """
    Extract article number from Điều text
    Example: "Điều 125. Hình thức..." -> 125
    """
    import re
    match = re.search(r'Điều\s+(\d+)', dieu_text)
    if match:
        return int(match.group(1))
    return 0


def collection_exists(collection_name):
    """Check if a collection exists in Qdrant Cloud"""
    collections = client.get_collections().collections
    return any(col.name == collection_name for col in collections)


def delete_collection_if_exists(collection_name):
    """Delete collection if it exists"""
    if collection_exists(collection_name):
        print(f"🗑️  Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        print(f"✅ Deleted collection '{collection_name}'")
        return True
    return False


def create_collection(collection_name):
    """Create a new collection"""
    print(f"📦 Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"✅ Created collection '{collection_name}'")


# ========== FAQ COLLECTION FUNCTION ==========

def recreate_faq_collection(force=False):
    """
    Create/Recreate FAQ collection with fresh embeddings in Qdrant Cloud

    Args:
        force: If True, delete existing collection and recreate

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("🔄 FAQ COLLECTION SETUP")
    print("="*80)

    collection_name = "faq_collection"

    try:
        # Load FAQ data
        print("📂 Loading FAQ data from faq (1).json...")
        try:
            with open("faq (1).json", "r", encoding="utf-8") as f:
                faq_data = json.load(f)
        except FileNotFoundError:
            print("❌ Error: faq (1).json file not found!")
            print("Please ensure the file exists in the current directory")
            return False

        faq_items = faq_data.get("meta", [])
        print(f"✅ Loaded {len(faq_items)} FAQ items")

        # Check if collection exists
        if collection_exists(collection_name):
            if force:
                delete_collection_if_exists(collection_name)
            else:
                print(f"✅ Collection '{collection_name}' already exists")
                count = client.get_collection(collection_name).points_count
                print(f"   Points in collection: {count}")
                print("💡 Set force=True to recreate with fresh embeddings")
                print("="*80)
                return True

        # Create collection
        create_collection(collection_name)

        # Add FAQ documents with fresh embeddings
        print(f"📄 Adding {len(faq_items)} FAQ documents...")
        points = []

        for idx, item in enumerate(tqdm(faq_items, desc="Processing FAQ items")):
            question = item.get("Câu hỏi", "")
            answer = item.get("Trả lời", "")

            if not question or not answer:
                continue

            # Combine question and answer for embedding
            combined_text = f"Câu hỏi: {question} Trả lời: {answer}"

            # Generate embedding
            vector = embeddings.embed_query(question)

            # Create payload with Vietnamese characters (Qdrant Cloud supports them)
            payload = {
                "Câu_hỏi": question,
                "Trả_lời": answer,
                "combined_text": combined_text,
                "source": "faq"
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            points.append(point)

            # Upload in batches of 100
            if len(points) >= 100:
                client.upsert(collection_name=collection_name, points=points)
                points = []

        # Upload remaining points
        if points:
            client.upsert(collection_name=collection_name, points=points)

        # Verify
        count = client.get_collection(collection_name).points_count
        print(f"✅ Verified: Collection has {count} points")
        print("="*80)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return False


# ========== LAW COLLECTION FUNCTION ==========

def recreate_law_collection(force=False):
    """
    Create/Recreate law collection with fresh embeddings in Qdrant Cloud
    Uses GPT-3.5-turbo to generate summaries before embedding

    Args:
        force: If True, delete existing collection and recreate

    Returns:
        bool: True if successful, False otherwise
    """
    print("\n" + "="*80)
    print("🔄 LAW COLLECTION SETUP")
    print("="*80)

    collection_name = "law_collection"

    try:
        # Load law data
        print("📂 Loading law data from law.json...")
        try:
            with open("law.json", "r", encoding="utf-8") as f:
                law_data = json.load(f)
        except FileNotFoundError:
            print("❌ Error: law.json file not found!")
            print("Please ensure the file exists in the current directory")
            return False

        law_items = law_data.get("meta", [])
        print(f"✅ Loaded {len(law_items)} law articles")

        # Check if collection exists
        if collection_exists(collection_name):
            if force:
                delete_collection_if_exists(collection_name)
            else:
                print(f"✅ Collection '{collection_name}' already exists")
                count = client.get_collection(collection_name).points_count
                print(f"   Points in collection: {count}")
                print("💡 Set force=True to recreate with fresh embeddings")
                print("="*80)
                return True

        # Step 1: Convert to Document objects
        print("📄 Converting to Document objects...")
        docs = []
        for item in law_items:
            dieu_text = item.get("Điều", "")
            metadata = {
                "Dieu": dieu_text,
                "Dieu_Number": extract_dieu_number(dieu_text),
                "Chuong": item.get("Chương", ""),
                "Muc": item.get("Mục", ""),
                "Pages": item.get("Pages", "")
            }

            doc = Document(
                page_content=item.get("Text", ""),
                metadata=metadata
            )
            docs.append(doc)

        print(f"✅ Created {len(docs)} Document objects")

        # Step 2: Generate summaries using GPT-3.5-turbo
        print("🤖 Generating summaries with GPT-3.5-turbo (this may take a while)...")

        query_str = """Hãy cung cấp một bản tóm tắt chi tiết bằng tiếng Việt về quy định pháp luật Việt Nam này, bao gồm:
- Các yêu cầu hoặc quy định pháp lý chính được nêu ra
- Những cá nhân, tổ chức hoặc đối tượng nào chịu sự điều chỉnh của quy định này
- Các nghĩa vụ, quyền hoặc thủ tục quan trọng được quy định
- Các điều kiện, ngoại lệ hoặc yêu cầu cụ thể đáng chú ý (nếu có)
- Mục đích hoặc phạm vi tổng thể của quy định pháp luật này

Vui lòng trình bày bản tóm tắt thành 3-4 đoạn văn bằng tiếng Việt, bảo đảm vừa đầy đủ vừa dễ đọc.
"""

        # Create chain for summarization
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template(f"{query_str}\n\nNội dung văn bản:\n\n{{doc}}")
            | ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, max_retries=1)
            | StrOutputParser()
        )

        # Batch process documents
        summaries = chain.batch(docs, {"max_concurrency": 5})
        print(f"✅ Generated {len(summaries)} summaries")

        # Create collection
        create_collection(collection_name)

        # Create payload index for Dieu_Number to enable filtering
        print(f"🔧 Creating payload index for metadata.Dieu_Number field...")
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.Dieu_Number",
                field_schema=PayloadSchemaType.INTEGER
            )
            print(f"✅ Created integer index for metadata.Dieu_Number")
        except Exception as e:
            print(f"⚠️  Warning: Could not create index for metadata.Dieu_Number: {e}")

        # Create payload index for Muc (Section)
        print(f"🔧 Creating payload index for metadata.Muc field...")
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.Muc",
                field_schema=PayloadSchemaType.TEXT
            )
            print(f"✅ Created TEXT index for metadata.Muc")
        except Exception as e:
            print(f"⚠️  Warning: Could not create index for metadata.Muc: {e}")

        # Create payload index for Chuong (Chapter)
        print(f"🔧 Creating payload index for metadata.Chuong field...")
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.Chuong",
                field_schema=PayloadSchemaType.TEXT
            )
            print(f"✅ Created TEXT index for metadata.Chuong")
        except Exception as e:
            print(f"⚠️  Warning: Could not create index for metadata.Chuong: {e}")

        # Step 3: Create embeddings from summaries and upload
        print(f"🚀 Creating embeddings from summaries and uploading...")
        points = []

        for doc, summary in tqdm(zip(docs, summaries), total=len(docs), desc="Processing documents"):
            if not summary or not summary.strip():
                print(f"⚠️  Skipping empty summary for {doc.metadata.get('Dieu', 'unknown')}")
                continue

            # Generate embedding from summary (not original text)
            vector = embeddings.embed_query(summary)

            # Create payload with nested metadata structure to match app's query format
            # The app uses QdrantTranslator(metadata_key="metadata") so fields must be under "metadata"
            payload = {
                "metadata": {
                    "Dieu": doc.metadata.get("Dieu", ""),
                    "Dieu_Number": doc.metadata.get("Dieu_Number", 0),
                    "Chuong": doc.metadata.get("Chuong", ""),
                    "Muc": doc.metadata.get("Muc", ""),
                    "Pages": doc.metadata.get("Pages", "")
                },
                "page_content": doc.page_content,  # ✅ REQUIRED: Main content for retrieval
                "source": "law",
                "original_text": doc.page_content,  # Keep original text for reference
                "summary": summary  # Full detailed summary
            }

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
            points.append(point)

            # Upload in batches of 100
            if len(points) >= 100:
                client.upsert(collection_name=collection_name, points=points)
                points = []

        # Upload remaining points
        if points:
            client.upsert(collection_name=collection_name, points=points)

        # Verify
        count = client.get_collection(collection_name).points_count
        print(f"✅ Verified: Collection has {count} points")
        print("="*80)

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("="*80)
        return False


# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 QDRANT CLOUD UPLOAD SCRIPT")
    print("="*80)
    print("\nThis script will upload FAQ and Law data to Qdrant Cloud")
    print("\nOptions:")
    print("1. Upload FAQ collection only")
    print("2. Upload Law collection only")
    print("3. Upload both collections")
    print("4. Recreate FAQ collection (force)")
    print("5. Recreate Law collection (force)")
    print("6. Recreate both collections (force)")
    print("\n" + "="*80)

    choice = input("\nEnter your choice (1-6): ").strip()

    if choice == "1":
        success = recreate_faq_collection(force=False)
        if success:
            print("\n✅ FAQ collection setup complete!")

    elif choice == "2":
        success = recreate_law_collection(force=False)
        if success:
            print("\n✅ Law collection setup complete!")

    elif choice == "3":
        success_faq = recreate_faq_collection(force=False)
        success_law = recreate_law_collection(force=False)
        if success_faq and success_law:
            print("\n✅ Both collections setup complete!")
        else:
            print("\n⚠️ Some collections failed to setup")

    elif choice == "4":
        success = recreate_faq_collection(force=True)
        if success:
            print("\n✅ FAQ collection recreated successfully!")

    elif choice == "5":
        success = recreate_law_collection(force=True)
        if success:
            print("\n✅ Law collection recreated successfully!")

    elif choice == "6":
        success_faq = recreate_faq_collection(force=True)
        success_law = recreate_law_collection(force=True)
        if success_faq and success_law:
            print("\n✅ Both collections recreated successfully!")
        else:
            print("\n⚠️ Some collections failed to recreate")

    else:
        print("❌ Invalid choice. Exiting.")
        exit(1)

    print("\n" + "="*80)
    print("🎉 UPLOAD COMPLETE!")
    print("="*80)
    print("\n📊 Summary:")

    # Show collection info
    try:
        if choice in ["1", "3", "4", "6"]:
            faq_info = client.get_collection("faq_collection")
            print(f"\n📚 FAQ Collection:")
            print(f"   - Points count: {faq_info.points_count}")
            print(f"   - Vector size: {faq_info.config.params.vectors.size}")

        if choice in ["2", "3", "5", "6"]:
            law_info = client.get_collection("law_collection")
            print(f"\n⚖️  Law Collection:")
            print(f"   - Points count: {law_info.points_count}")
            print(f"   - Vector size: {law_info.config.params.vectors.size}")
    except Exception as e:
        print(f"\n⚠️  Could not retrieve collection info: {e}")

    print("\n👉 You can now use the chatbot with data from Qdrant Cloud!")
    print("="*80 + "\n")
