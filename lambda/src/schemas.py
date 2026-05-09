"""データモデル定義"""

from typing import Any

from pydantic import BaseModel, Field


class RequestBody(BaseModel):
    """API リクエストボディ"""

    input_text: str
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    system_instruction: str | None = None


class EstimatedCostInfo(BaseModel):
    """コスト見積もり情報"""

    estimatedCost: float
    currency: str

    class Config:
        extra = "allow"


class UsageSummaryEntry(BaseModel):
    """利用状況サマリーエントリ"""

    modelVersion: str
    requestCount: int
    tokens: dict[str, int]
    estimatedCostInfo: EstimatedCostInfo | None = None


class ResponseBody(BaseModel):
    """API レスポンスボディ"""

    outputs: str
    usageMetadata: list[UsageSummaryEntry] | None = None


class ErrorResponse(BaseModel):
    """エラーレスポンス"""

    error: str
