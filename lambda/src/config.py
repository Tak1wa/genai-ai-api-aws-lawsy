"""AWS版 Lawsy の設定管理モジュール"""

import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    """アプリケーション設定"""

    # AWS リージョン
    region: str

    # Bedrock モデル設定
    model_id: str
    embedding_model_id: str

    # S3 Vectors 設定
    vector_bucket_arn: str
    vector_index_name: str

    # S3 データバケット
    data_bucket_name: str

    # 生成パラメータ
    temperature: float
    max_tokens: int
    top_p: float


def load_config() -> AppConfig:
    """環境変数から設定を読み込む"""
    return AppConfig(
        region=os.environ.get("AWS_REGION", "ap-northeast-1"),
        model_id=os.environ.get("MODEL_ID", "anthropic.claude-sonnet-4-20250514-v1:0"),
        embedding_model_id=os.environ.get(
            "EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
        ),
        vector_bucket_arn=os.environ.get("VECTOR_BUCKET_ARN", ""),
        vector_index_name=os.environ.get("VECTOR_INDEX_NAME", "laws-index"),
        data_bucket_name=os.environ.get("DATA_BUCKET_NAME", "lawsy-aws-data-PLACEHOLDER"),
        temperature=float(os.environ.get("TEMPERATURE", "0.0")),
        max_tokens=int(os.environ.get("MAX_TOKENS", "8192")),
        top_p=float(os.environ.get("TOP_P", "1.0")),
    )
