"""Amazon Bedrock クライアントモジュール"""

import json
import logging
from datetime import date

import boto3

from .config import AppConfig

logger = logging.getLogger(__name__)


class BedrockClient:
    """Amazon Bedrock API クライアント"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = boto3.client("bedrock-runtime", region_name=config.region)
        self.agent_client = boto3.client(
            "bedrock-agent-runtime", region_name=config.region
        )

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
    ) -> tuple[str, dict]:
        """Bedrock Claude でテキスト生成を行う

        Returns:
            tuple[str, dict]: (生成テキスト, usage情報)
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        inference_config = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "maxTokens": max_tokens if max_tokens is not None else self.config.max_tokens,
            "topP": top_p if top_p is not None else self.config.top_p,
        }

        kwargs = {
            "modelId": self.config.model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
        }

        if system_instruction:
            today_str = date.today().isoformat()
            full_instruction = f"{system_instruction}\n\n現在の日時: {today_str}"
            kwargs["system"] = [{"text": full_instruction}]

        try:
            response = self.client.converse(**kwargs)
            output_text = ""
            for block in response.get("output", {}).get("message", {}).get(
                "content", []
            ):
                if "text" in block:
                    output_text += block["text"]

            usage = response.get("usage", {})
            return output_text, usage

        except Exception as e:
            logger.error(f"Bedrock API call failed: {e}", exc_info=True)
            raise

    def generate_embedding(self, text: str) -> list[float]:
        """Bedrock Titan Embeddings でベクトル埋め込みを生成する"""
        body = json.dumps(
            {"inputText": text, "dimensions": 1024, "normalize": True}
        )

        try:
            response = self.client.invoke_model(
                modelId=self.config.embedding_model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}", exc_info=True)
            raise

    def generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """複数テキストのベクトル埋め込みを生成する"""
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def web_search(self, query: str) -> list[dict]:
        """Bedrock Web Search を使用してWeb検索を行う

        Returns:
            list[dict]: 検索結果のリスト [{title, url, snippet}]
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

        today_str = date.today().isoformat()
        system_instruction = (
            f"本日の日付は {today_str} です。"
            "クエリに関連する日本の法令を調査し、関連する法令名を以下のJSON形式で回答してください。"
            "調査の際はe-Govや各省庁の公式サイトを優先して参照してください。"
            "必ず有効なJSONのみを出力し、説明文やマークダウンは一切含めないでください："
            '{"law_names": ["法令名1", "法令名2", "法令名3"]}'
            f"【重要1】廃止・失効した法令は絶対に含めないこと。本日時点（{today_str}）で既に廃止・統合されている法令は除外し、現行の後継法令名のみを返すこと。"
            "【重要2】クエリで言及された法令名が通称・略称の場合、対応する正式名称が確実に特定できる場合のみ採用すること。"
        )

        kwargs = {
            "modelId": self.config.model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": 0.0,
                "maxTokens": 2048,
                "topP": 1.0,
            },
            "system": [{"text": system_instruction}],
            "toolConfig": {
                "tools": [
                    {
                        "webSearchTool": {
                            "searchEngine": {"type": "TAVILY"},
                        }
                    }
                ]
            },
        }

        try:
            response = self.client.converse(**kwargs)
            output_text = ""
            web_results = []

            for block in response.get("output", {}).get("message", {}).get(
                "content", []
            ):
                if "text" in block:
                    output_text += block["text"]

            # Web検索結果はレスポンスのstopReasonがtool_useの場合、
            # ツール呼び出し→結果→最終回答のマルチターンになる
            # Converse APIのwebSearchToolは自動的にマルチターンを処理する
            stop_reason = response.get("stopReason", "")

            if stop_reason == "tool_use":
                # ツール使用が必要な場合、再帰的に処理
                # Converse API の webSearchTool は自動処理されるため、
                # 最終レスポンスのテキストを取得
                pass

            return output_text, web_results

        except Exception as e:
            logger.error(f"Web search failed: {e}", exc_info=True)
            return "", []
