"""AWS版 Lawsy データパイプライン

法令XMLを解析し、S3 Vectorsにベクトルデータを投入する。

Usage:
    python run_entire_pipeline.py \
        --region ap-northeast-1 \
        --vector-bucket-name your-vector-bucket \
        --vector-index-name laws-index \
        --data-bucket-name your-data-bucket \
        --xml-dir ./xml_files \
        --date-tag 20250509
"""

import argparse
import json
import sys
import time

import boto3
from tqdm import tqdm

from parse_law_xml import parse_all_xml_files


def create_vector_bucket_if_not_exists(s3vectors_client, bucket_name: str) -> str:
    """S3 Vectors バケットを作成する（存在しない場合）"""
    try:
        response = s3vectors_client.list_vector_buckets()
        for bucket in response.get("vectorBuckets", []):
            if bucket["name"] == bucket_name:
                print(f"INFO: Vector bucket '{bucket_name}' already exists.", file=sys.stderr)
                return bucket["vectorBucketArn"]
    except Exception:
        pass

    print(f"INFO: Creating vector bucket '{bucket_name}'...", file=sys.stderr)
    response = s3vectors_client.create_vector_bucket(vectorBucketName=bucket_name)
    arn = response["vectorBucketArn"]
    print(f"SUCCESS: Vector bucket created: {arn}", file=sys.stderr)
    return arn


def create_vector_index_if_not_exists(
    s3vectors_client, vector_bucket_arn: str, index_name: str, dimension: int = 1024
) -> None:
    """S3 Vectors インデックスを作成する（存在しない場合）"""
    try:
        s3vectors_client.get_vector_index(
            vectorBucketArn=vector_bucket_arn, indexName=index_name
        )
        print(f"INFO: Vector index '{index_name}' already exists.", file=sys.stderr)
        return
    except s3vectors_client.exceptions.NotFoundException:
        pass
    except Exception as e:
        if "not found" in str(e).lower() or "NotFoundException" in str(type(e)):
            pass
        else:
            raise

    print(f"INFO: Creating vector index '{index_name}' (dimension={dimension})...", file=sys.stderr)
    s3vectors_client.create_vector_index(
        vectorBucketArn=vector_bucket_arn,
        indexName=index_name,
        dimension=dimension,
        distanceMetric="cosine",
        metadata={
            "config": [
                {"key": "law_num", "dataType": "str"},
                {"key": "law_id", "dataType": "str"},
                {"key": "law_title", "dataType": "str"},
                {"key": "unique_anchor", "dataType": "str", "filterable": False},
                {"key": "anchor", "dataType": "str", "filterable": False},
                {"key": "article_summary", "dataType": "str", "filterable": False},
                {"key": "content", "dataType": "str", "filterable": False},
            ]
        },
    )
    print(f"SUCCESS: Vector index '{index_name}' created.", file=sys.stderr)

    # インデックスがアクティブになるまで待機
    print("INFO: Waiting for vector index to become active...", file=sys.stderr)
    while True:
        response = s3vectors_client.get_vector_index(
            vectorBucketArn=vector_bucket_arn, indexName=index_name
        )
        status = response.get("status", "")
        if status == "ACTIVE":
            print("SUCCESS: Vector index is active.", file=sys.stderr)
            break
        print(f"  Status: {status}, waiting...", file=sys.stderr)
        time.sleep(5)


def generate_embedding(bedrock_client, text: str, model_id: str = "amazon.titan-embed-text-v2:0") -> list[float]:
    """Bedrock Titan Embeddings でベクトルを生成する"""
    body = json.dumps({"inputText": text, "dimensions": 1024, "normalize": True})
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    response_body = json.loads(response["body"].read())
    return response_body["embedding"]


def upload_law_master_vectors(
    s3vectors_client,
    bedrock_client,
    vector_bucket_arn: str,
    index_name: str,
    jsonl_file: str,
):
    """法令マスタ（法令名ごとに1ベクトル）をS3 Vectorsに投入する

    Google Cloud版の app_laws_master に相当。
    法令名のembeddingを生成し、法令名検索用のベクトルとして登録する。
    """
    print("INFO: Building law master vectors...", file=sys.stderr)

    # JSONLから法令名の一覧を抽出（重複排除）
    law_master = {}  # law_num -> {law_title, law_id}
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            law_num = row["law_num"]
            if law_num not in law_master:
                law_master[law_num] = {
                    "law_title": row["law_title"],
                    "law_id": row["law_id"],
                }

    print(f"INFO: Found {len(law_master)} unique laws.", file=sys.stderr)

    # バッチでベクトルを投入
    batch = []
    batch_size = 10

    for law_num, info in tqdm(law_master.items(), desc="Generating law master embeddings"):
        law_title = info["law_title"]
        try:
            embedding = generate_embedding(bedrock_client, law_title)
        except Exception as e:
            print(f"WARNING: Embedding failed for '{law_title}': {e}", file=sys.stderr)
            continue

        vector_data = {
            "key": f"master_{law_num}",
            "data": {"float32": embedding},
            "metadata": {
                "law_num": law_num,
                "law_id": info["law_id"],
                "law_title": law_title,
            },
        }
        batch.append(vector_data)

        if len(batch) >= batch_size:
            _put_vectors_batch(s3vectors_client, vector_bucket_arn, index_name, batch)
            batch = []

    if batch:
        _put_vectors_batch(s3vectors_client, vector_bucket_arn, index_name, batch)

    print(f"SUCCESS: Uploaded {len(law_master)} law master vectors.", file=sys.stderr)


def upload_article_vectors(
    s3vectors_client,
    vector_bucket_arn: str,
    index_name: str,
    jsonl_file: str,
):
    """条文データをS3 Vectorsに投入する（メタデータとして格納）

    Google Cloud版の app_laws_for_indexing に相当。
    条文の内容はメタデータとして格納し、法令名ベクトル検索でマッチした法令の
    全条文をメタデータフィルタで取得する。
    """
    print("INFO: Uploading article data to S3 Vectors...", file=sys.stderr)

    batch = []
    batch_size = 10
    total_uploaded = 0

    # ダミーのゼロベクトル（条文はベクトル検索対象ではなく、メタデータフィルタで取得）
    # S3 Vectorsではベクトルが必須のため、法令マスタのembeddingを流用するか
    # ダミーベクトルを使用する
    dummy_vector = [0.0] * 1024

    with open(jsonl_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Uploading article vectors"):
        row = json.loads(line)

        # メタデータサイズ制限を考慮してcontentを切り詰め
        content = row.get("content", "")
        if len(content) > 40000:
            content = content[:40000]

        metadata = {
            "law_num": row["law_num"],
            "law_id": row["law_id"],
            "law_title": row["law_title"],
            "unique_anchor": row["unique_anchor"],
            "anchor": row.get("anchor") or "",
            "article_summary": row.get("article_summary") or "",
            "content": content,
        }

        vector_data = {
            "key": row["unique_anchor"],
            "data": {"float32": dummy_vector},
            "metadata": metadata,
        }
        batch.append(vector_data)

        if len(batch) >= batch_size:
            _put_vectors_batch(s3vectors_client, vector_bucket_arn, index_name, batch)
            total_uploaded += len(batch)
            batch = []

    if batch:
        _put_vectors_batch(s3vectors_client, vector_bucket_arn, index_name, batch)
        total_uploaded += len(batch)

    print(f"SUCCESS: Uploaded {total_uploaded} article vectors.", file=sys.stderr)


def _put_vectors_batch(s3vectors_client, vector_bucket_arn: str, index_name: str, batch: list):
    """ベクトルをバッチで投入する"""
    try:
        s3vectors_client.put_vectors(
            vectorBucketArn=vector_bucket_arn,
            indexName=index_name,
            vectors=batch,
        )
    except Exception as e:
        print(f"WARNING: put_vectors failed: {e}", file=sys.stderr)
        # 個別にリトライ
        for vector in batch:
            try:
                s3vectors_client.put_vectors(
                    vectorBucketArn=vector_bucket_arn,
                    indexName=index_name,
                    vectors=[vector],
                )
            except Exception as e2:
                print(
                    f"ERROR: Failed to put vector key={vector['key']}: {e2}",
                    file=sys.stderr,
                )


def main():
    parser = argparse.ArgumentParser(description="AWS版 Lawsy データパイプライン")
    parser.add_argument("--region", default="ap-northeast-1", help="AWS リージョン")
    parser.add_argument("--vector-bucket-name", required=True, help="S3 Vectors バケット名")
    parser.add_argument("--vector-index-name", default="laws-index", help="ベクトルインデックス名")
    parser.add_argument("--data-bucket-name", required=True, help="データ保管用S3バケット名")
    parser.add_argument("--xml-dir", required=True, help="法令XMLファイルのディレクトリ")
    parser.add_argument("--date-tag", required=True, help="日付タグ (例: 20250509)")
    parser.add_argument("--embedding-model-id", default="amazon.titan-embed-text-v2:0", help="Embeddingモデル")

    args = parser.parse_args()

    # クライアント初期化
    s3vectors_client = boto3.client("s3vectors", region_name=args.region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=args.region)
    s3_client = boto3.client("s3", region_name=args.region)

    # Step 1: XMLパース → JSONL
    print("=" * 60, file=sys.stderr)
    print("Step 1: Parsing law XML files...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    jsonl_file = f"laws_{args.date_tag}.jsonl"
    total_rows = parse_all_xml_files(args.xml_dir, jsonl_file)
    if total_rows == 0:
        print("ERROR: No data generated. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Step 2: JSONLをS3にアップロード（バックアップ）
    print("=" * 60, file=sys.stderr)
    print("Step 2: Uploading JSONL to S3...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    try:
        # バケットが存在しない場合は作成
        try:
            s3_client.head_bucket(Bucket=args.data_bucket_name)
        except Exception:
            s3_client.create_bucket(
                Bucket=args.data_bucket_name,
                CreateBucketConfiguration={"LocationConstraint": args.region},
            )
            print(f"INFO: Created S3 bucket: {args.data_bucket_name}", file=sys.stderr)

        s3_client.upload_file(jsonl_file, args.data_bucket_name, f"laws/{jsonl_file}")
        print(f"SUCCESS: Uploaded to s3://{args.data_bucket_name}/laws/{jsonl_file}", file=sys.stderr)
    except Exception as e:
        print(f"WARNING: S3 upload failed (non-fatal): {e}", file=sys.stderr)

    # Step 3: S3 Vectors バケット・インデックス作成
    print("=" * 60, file=sys.stderr)
    print("Step 3: Setting up S3 Vectors...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    vector_bucket_arn = create_vector_bucket_if_not_exists(
        s3vectors_client, args.vector_bucket_name
    )
    create_vector_index_if_not_exists(
        s3vectors_client, vector_bucket_arn, args.vector_index_name
    )

    # Step 4: 法令マスタベクトルの投入
    print("=" * 60, file=sys.stderr)
    print("Step 4: Uploading law master vectors...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    upload_law_master_vectors(
        s3vectors_client,
        bedrock_client,
        vector_bucket_arn,
        args.vector_index_name,
        jsonl_file,
    )

    # Step 5: 条文データの投入
    print("=" * 60, file=sys.stderr)
    print("Step 5: Uploading article data...", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    upload_article_vectors(
        s3vectors_client,
        vector_bucket_arn,
        args.vector_index_name,
        jsonl_file,
    )

    print("=" * 60, file=sys.stderr)
    print("Pipeline completed successfully!", file=sys.stderr)
    print(f"  Vector Bucket ARN: {vector_bucket_arn}", file=sys.stderr)
    print(f"  Vector Index Name: {args.vector_index_name}", file=sys.stderr)
    print(f"  Total articles: {total_rows}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)


if __name__ == "__main__":
    main()
