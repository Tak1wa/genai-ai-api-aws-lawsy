"""Bedrock API 利用状況トラッカー"""

from collections import defaultdict
from typing import Any

from .schemas import EstimatedCostInfo

# Bedrock Claude 料金情報 (USD per 1M tokens)
MODEL_PRICING = {
    "anthropic.claude-sonnet-4-20250514-v1:0": {
        "currency": "USD",
        "pricing_unit": 1_000_000,
        "input_price": 3.0,
        "output_price": 15.0,
    },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {
        "currency": "USD",
        "pricing_unit": 1_000_000,
        "input_price": 3.0,
        "output_price": 15.0,
    },
    "amazon.titan-embed-text-v2:0": {
        "currency": "USD",
        "pricing_unit": 1_000_000,
        "input_price": 0.02,
        "output_price": 0.0,
    },
}


class UsageTracker:
    """Bedrock API の利用状況を追跡するクラス"""

    def __init__(self):
        self.usages: list[tuple[str, dict]] = []

    def add_usage(self, model_id: str, usage: dict):
        """利用情報を追加する

        Args:
            model_id: モデルID
            usage: Bedrock API の usage レスポンス
        """
        if usage:
            self.usages.append((model_id, usage))

    def get_usage_summary(self) -> list[dict[str, Any]]:
        """蓄積された利用情報からサマリーを生成する"""
        summary_data: dict[str, dict] = defaultdict(
            lambda: {"requestCount": 0, "tokens": defaultdict(int)}
        )

        for model_id, usage in self.usages:
            summary_data[model_id]["requestCount"] += 1
            if "inputTokens" in usage:
                summary_data[model_id]["tokens"]["inputTokens"] += usage["inputTokens"]
            if "outputTokens" in usage:
                summary_data[model_id]["tokens"]["outputTokens"] += usage["outputTokens"]
            if "totalTokens" in usage:
                summary_data[model_id]["tokens"]["totalTokens"] += usage["totalTokens"]

        output_summary = []
        for model_id, data in summary_data.items():
            tokens_dict = dict(data["tokens"])
            estimated_cost = self._calculate_cost(model_id, tokens_dict)

            entry = {
                "modelVersion": model_id,
                "requestCount": data["requestCount"],
                "tokens": tokens_dict,
            }
            if estimated_cost:
                entry["estimatedCostInfo"] = estimated_cost.dict()

            output_summary.append(entry)

        return output_summary

    def _calculate_cost(
        self, model_id: str, tokens: dict[str, int]
    ) -> EstimatedCostInfo | None:
        """コストを計算する"""
        pricing = MODEL_PRICING.get(model_id)
        if not pricing:
            return None

        unit = pricing["pricing_unit"]
        input_tokens = tokens.get("inputTokens", 0)
        output_tokens = tokens.get("outputTokens", 0)

        input_cost = (input_tokens / unit) * pricing["input_price"]
        output_cost = (output_tokens / unit) * pricing["output_price"]
        total_cost = input_cost + output_cost

        return EstimatedCostInfo(
            estimatedCost=total_cost,
            currency=pricing["currency"],
            inputTokens=input_tokens,
            outputTokens=output_tokens,
        )
