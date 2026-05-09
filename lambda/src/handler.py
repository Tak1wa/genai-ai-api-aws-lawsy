"""AWS Lambda ハンドラー（エントリーポイント）"""

import json
import logging
import sys

from .bedrock_client import BedrockClient
from .config import AppConfig, load_config
from .law_report_pipeline import generate_law_report
from .retrieval_s3vectors import S3VectorsRetriever
from .schemas import ResponseBody

# ロギング設定
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# グローバル変数（Lambda コールドスタート時に初期化）
app_config: AppConfig | None = None
bedrock_client: BedrockClient | None = None
retriever: S3VectorsRetriever | None = None

try:
    logger.info("アプリケーションの初期化を開始します...")
    app_config = load_config()
    bedrock_client = BedrockClient(app_config)
    retriever = S3VectorsRetriever(
        vector_bucket_arn=app_config.vector_bucket_arn,
        vector_index_name=app_config.vector_index_name,
        bedrock_client=bedrock_client,
        data_bucket_name=app_config.data_bucket_name,
    )
    logger.info("アプリケーションの初期化が完了しました。")
except Exception as e:
    logger.critical(f"初期化中に致命的なエラーが発生しました: {e}", exc_info=True)
    app_config = None
    bedrock_client = None
    retriever = None


def _create_response(body: dict, status_code: int) -> dict:
    """API Gateway 用レスポンスを生成する"""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,x-api-key",
            "Access-Control-Allow-Methods": "POST,OPTIONS",
        },
        "body": json.dumps(body, ensure_ascii=False),
    }


def lambda_handler(event, context):
    """Lambda エントリーポイント"""
    if not app_config or not bedrock_client or not retriever:
        return _create_response(
            {"error": "Internal Server Error: Application not initialized."}, 500
        )

    # OPTIONS リクエスト（CORS プリフライト）
    http_method = event.get("httpMethod") or event.get("requestContext", {}).get(
        "http", {}
    ).get("method", "")
    if http_method == "OPTIONS":
        return _create_response({}, 204)

    if http_method != "POST":
        return _create_response({"error": "Method not allowed"}, 405)

    try:
        # リクエストボディの解析
        body = event.get("body", "")
        if isinstance(body, str):
            payload = json.loads(body) if body else {}
        else:
            payload = body or {}

        # input_text の取得（源内Web互換: inputs.input_text）
        input_text = payload.get("inputs", {}).get("input_text")
        if not input_text:
            # フラットな形式もサポート
            input_text = payload.get("input_text")

        if not input_text:
            return _create_response(
                {"error": 'Invalid payload. "inputs.input_text" is required.'}, 400
            )

        logger.info(f"処理開始: クエリ='{input_text}'")

        # レポート生成
        final_report_content, usage_summary = generate_law_report(
            input_text, bedrock_client, app_config, retriever
        )

        # レスポンス構築
        response_body = ResponseBody(
            outputs=final_report_content, usageMetadata=usage_summary
        )
        return _create_response(response_body.dict(exclude_none=True), 200)

    except json.JSONDecodeError:
        return _create_response({"error": "Invalid JSON"}, 400)
    except Exception as e:
        logger.error(f"エラー発生: {e!s}", exc_info=True)
        return _create_response({"error": "Internal server error"}, 500)
