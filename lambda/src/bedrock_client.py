"""Amazon Bedrock クライアントモジュール"""

import html
import json
import logging
import re
from datetime import date
from urllib.parse import urlparse

import boto3
import requests

from .config import AppConfig

logger = logging.getLogger(__name__)


class BedrockClient:
    """Amazon Bedrock API クライアント"""

    def __init__(self, config: AppConfig):
        self.config = config
        self.client = boto3.client("bedrock-runtime", region_name=config.region)

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
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        inference_config = {
            "temperature": temperature if temperature is not None else self.config.temperature,
            "maxTokens": max_tokens if max_tokens is not None else self.config.max_tokens,
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

    def generate_text_with_web_search(
        self,
        prompt: str,
        system_instruction: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict, list[dict]]:
        """Bedrock Claude + Web Search でテキスト生成を行う

        Google Cloud版の Web Grounding に相当。
        Converse API の webSearchTool を使用してリアルタイムWeb検索を行う。

        Returns:
            tuple[str, dict, list[dict]]: (生成テキスト, usage情報, web検索結果)
        """
        messages = [{"role": "user", "content": [{"text": prompt}]}]

        inference_config = {
            "temperature": temperature if temperature is not None else 0.0,
            "maxTokens": max_tokens if max_tokens is not None else 2048,
        }

        kwargs = {
            "modelId": self.config.model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
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

        if system_instruction:
            today_str = date.today().isoformat()
            full_instruction = f"{system_instruction}\n\n現在の日時: {today_str}"
            kwargs["system"] = [{"text": full_instruction}]

        try:
            response = self.client.converse(**kwargs)
            output_text = ""
            web_results = []

            # レスポンスからテキストとWeb検索結果を抽出
            output_message = response.get("output", {}).get("message", {})
            for block in output_message.get("content", []):
                if "text" in block:
                    output_text += block["text"]

            # stopReason が tool_use の場合、マルチターンで処理
            stop_reason = response.get("stopReason", "")
            if stop_reason == "tool_use":
                # ツール呼び出し結果を含むマルチターン処理
                output_text, web_results = self._handle_web_search_tool_use(
                    response, messages, kwargs
                )

            usage = response.get("usage", {})
            return output_text, usage, web_results

        except Exception as e:
            logger.error(f"Web search generation failed: {e}", exc_info=True)
            # フォールバック: Web検索なしで生成
            logger.info("Falling back to generation without web search...")
            try:
                text, usage = self.generate_text(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return text, usage, []
            except Exception as e2:
                logger.error(f"Fallback also failed: {e2}")
                raise

    def _handle_web_search_tool_use(
        self, initial_response: dict, messages: list, kwargs: dict
    ) -> tuple[str, list[dict]]:
        """Web検索ツール使用のマルチターン処理

        Converse API が tool_use で停止した場合、ツール結果を返して最終回答を取得する。
        """
        web_results = []
        output_message = initial_response.get("output", {}).get("message", {})

        # アシスタントメッセージを追加
        messages.append(output_message)

        # ツール使用ブロックを処理
        tool_results = []
        for block in output_message.get("content", []):
            if "toolUse" in block:
                tool_use = block["toolUse"]
                tool_use_id = tool_use.get("toolUseId", "")
                # webSearchTool の結果はConverse APIが自動処理するため、
                # ここでは空の結果を返す
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": tool_use_id,
                            "content": [{"text": "検索完了"}],
                        }
                    }
                )

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            kwargs["messages"] = messages

            try:
                response = self.client.converse(**kwargs)
                output_text = ""
                for block in response.get("output", {}).get("message", {}).get(
                    "content", []
                ):
                    if "text" in block:
                        output_text += block["text"]
                return output_text, web_results
            except Exception as e:
                logger.error(f"Multi-turn web search failed: {e}")

        return "", web_results

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


def fetch_page_info(url: str, fallback_domain: str) -> tuple[str, str]:
    """URLからページタイトルを取得する（Google Cloud版の _fetch_page_info に相当）

    Returns:
        tuple[str, str]: (最終URL, ページタイトル)
    """
    fallback_url = f"https://{fallback_domain}" if fallback_domain else url
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=6,
            allow_redirects=True,
        )
        final_url = resp.url
        # UTF-8でデコードを試みる
        try:
            body = resp.content[:8192].decode("utf-8")
        except UnicodeDecodeError:
            encoding = resp.apparent_encoding or "shift_jis"
            body = resp.content[:8192].decode(encoding, errors="replace")

        m = re.search(r"<title[^>]*>(.*?)</title>", body, re.IGNORECASE | re.DOTALL)
        page_title = html.unescape(m.group(1).strip()) if m else fallback_domain
        return final_url, page_title
    except Exception:
        return fallback_url, fallback_domain
