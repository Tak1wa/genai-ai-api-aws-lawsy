"""S3 Vectors へのデータ投入スクリプト

法令マスタ（法令名ベクトル）と条文データをS3 Vectorsに投入する。
"""

import json
import sys
import time

import boto3
from tqdm import tqdm

VECTOR_BUCKET_NAME = "lawsy-aws-vectors"
INDEX_NAME = "laws-index"
REGION = "ap-northeast-1"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v2:0"


def generate_embedding(bedrock_client, text: str) -> list:
    """Titan Embeddings V2 でベクトルを生成"""
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    response = bedrock_client.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


def load_law_master(jsonl_file: str) -> dict:
    """JSONLから法令マスタ（法令名の重複排除リスト）を構築"""
    law_master = {}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            law_num = row["law_num"]
            if law_num not in law_master:
                law_master[law_num] = {
                    "law_title": row["law_title"],
                    "law_id": row["law_id"],
                }
    return law_master


def upload_law_master_vectors(jsonl_file: str):
    """法令マスタベクトルを投入"""
    s3vectors = boto3.client("s3vectors", region_name=REGION)
    bedrock = boto3.client("bedrock-runtime", region_name=REGION)

    law_master = load_law_master(jsonl_file)
    print(f"法令マスタ: {len(law_master)} 件", file=sys.stderr)

    batch = []
    batch_size = 5
    success_count = 0
    error_count = 0

    for law_num, info in tqdm(law_master.items(), desc="法令マスタ投入"):
        try:
            embedding = generate_embedding(bedrock, info["law_title"])
            vector = {
                "key": f"master_{law_num}",
                "data": {"float32": embedding},
                "metadata": {
                    "law_num": law_num,
                    "law_id": info["law_id"],
                    "law_title": info["law_title"],
                },
            }
            batch.append(vector)

            if len(batch) >= batch_size:
                s3vectors.put_vectors(
                    vectorBucketName=VECTOR_BUCKET_NAME,
                    indexName=INDEX_NAME,
                    vectors=batch,
                )
                success_count += len(batch)
                batch = []

        except Exception as e:
            error_count += 1
            if "ThrottlingException" in str(e) or "Too Many" in str(e):
                time.sleep(2)
            else:
                print(f"ERROR: {info['law_title']}: {e}", file=sys.stderr)
            continue

    if batch:
        try:
            s3vectors.put_vectors(
                vectorBucketName=VECTOR_BUCKET_NAME,
                indexName=INDEX_NAME,
                vectors=batch,
            )
            success_count += len(batch)
        except Exception as e:
            print(f"ERROR: final batch: {e}", file=sys.stderr)

    print(f"法令マスタ投入完了: 成功={success_count}, エラー={error_count}", file=sys.stderr)


def upload_article_data(jsonl_file: str):
    """条文データを投入（ベクトルはダミー、メタデータに条文内容を格納）"""
    s3vectors = boto3.client("s3vectors", region_name=REGION)

    # ダミーベクトル（条文はベクトル検索対象ではなく、メタデータフィルタで取得）
    dummy_vector = [0.0] * 1024

    batch = []
    batch_size = 10
    success_count = 0
    error_count = 0

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"条文データ: {len(lines)} 件", file=sys.stderr)

    for line in tqdm(lines, desc="条文データ投入"):
        row = json.loads(line)

        # S3 Vectors のメタデータサイズ制限を考慮
        content = row.get("content", "")
        if len(content.encode("utf-8")) > 40000:
            content = content[:10000]

        article_summary = row.get("article_summary", "") or ""
        if len(article_summary.encode("utf-8")) > 5000:
            article_summary = article_summary[:1000]

        metadata = {
            "law_num": row["law_num"],
            "law_id": row["law_id"],
            "law_title": row["law_title"],
        }

        # non-filterable metadata
        metadata["unique_anchor"] = row["unique_anchor"]
        metadata["anchor"] = row.get("anchor") or ""
        metadata["article_summary"] = article_summary
        metadata["content"] = content

        vector = {
            "key": row["unique_anchor"],
            "data": {"float32": dummy_vector},
            "metadata": metadata,
        }
        batch.append(vector)

        if len(batch) >= batch_size:
            try:
                s3vectors.put_vectors(
                    vectorBucketName=VECTOR_BUCKET_NAME,
                    indexName=INDEX_NAME,
                    vectors=batch,
                )
                success_count += len(batch)
            except Exception as e:
                error_count += len(batch)
                if "ThrottlingException" in str(e):
                    time.sleep(1)
                else:
                    print(f"ERROR: {e}", file=sys.stderr)
            batch = []

    if batch:
        try:
            s3vectors.put_vectors(
                vectorBucketName=VECTOR_BUCKET_NAME,
                indexName=INDEX_NAME,
                vectors=batch,
            )
            success_count += len(batch)
        except Exception as e:
            error_count += len(batch)
            print(f"ERROR: final batch: {e}", file=sys.stderr)

    print(f"条文データ投入完了: 成功={success_count}, エラー={error_count}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_to_s3vectors.py <jsonl_file> [master|articles|all]", file=sys.stderr)
        sys.exit(1)

    jsonl_file = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"

    if mode in ("master", "all"):
        upload_law_master_vectors(jsonl_file)

    if mode in ("articles", "all"):
        upload_article_data(jsonl_file)
