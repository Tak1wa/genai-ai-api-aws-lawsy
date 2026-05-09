"""Amazon S3 Vectors を利用した法令検索の実装"""

import logging

import boto3
from pydantic import BaseModel

from .bedrock_client import BedrockClient

logger = logging.getLogger(__name__)


# ---------------------------
# データモデル
# ---------------------------


class ArticleWithSummary(BaseModel):
    """条文概要付きデータ"""

    law_num: str
    law_id: str
    law_title: str
    unique_anchor: str
    article_summary: str | None = None
    content: str | None = None
    is_summary_only: bool = False


class FullArticle(BaseModel):
    """条文全文データ"""

    law_id: str
    title: str
    content: str
    unique_anchor: str
    anchor: str | None = None
    url: str


# ---------------------------
# S3 Vectors リトリーバー
# ---------------------------


class S3VectorsRetriever:
    """Amazon S3 Vectors を使用した法令検索"""

    def __init__(self, vector_bucket_arn: str, vector_index_name: str, bedrock_client: BedrockClient):
        self.vector_bucket_arn = vector_bucket_arn
        self.vector_index_name = vector_index_name
        self.bedrock_client = bedrock_client
        self.s3vectors_client = boto3.client(
            "s3vectors", region_name=bedrock_client.config.region
        )
        logger.info(
            f"S3VectorsRetriever initialized: bucket={vector_bucket_arn}, index={vector_index_name}"
        )

    def get_articles_by_nearest_law(
        self, law_names: list[str]
    ) -> list[ArticleWithSummary]:
        """法令名候補からベクトル検索で最近傍の法令条文を取得する

        各法令名に対してembeddingを生成し、S3 Vectorsで類似検索を行い、
        マッチした法令の全条文を返す。
        """
        if not law_names:
            return []

        all_articles = []
        matched_law_nums = set()

        for law_name in law_names:
            try:
                # 法令名のembeddingを生成
                query_embedding = self.bedrock_client.generate_embedding(law_name)

                # S3 Vectors でベクトル検索（法令マスタから最近傍を取得）
                response = self.s3vectors_client.query_vectors(
                    vectorBucketArn=self.vector_bucket_arn,
                    indexName=self.vector_index_name,
                    queryVector={"float32": query_embedding},
                    topK=1,
                    returnMetadata=True,
                    returnDistance=True,
                )

                vectors = response.get("vectors", [])
                if vectors:
                    vector = vectors[0]
                    metadata = vector.get("metadata", {})
                    law_num = metadata.get("law_num", "")
                    if law_num and law_num not in matched_law_nums:
                        matched_law_nums.add(law_num)
                        logger.info(
                            f"Matched law: '{law_name}' -> '{metadata.get('law_title', '')}' "
                            f"(distance={vector.get('distance', 'N/A')})"
                        )

            except Exception as e:
                logger.error(f"Vector search failed for '{law_name}': {e}")
                continue

        if not matched_law_nums:
            return []

        # マッチした法令番号の全条文を取得
        for law_num in matched_law_nums:
            try:
                articles = self._get_all_articles_for_law(law_num)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Failed to get articles for law_num={law_num}: {e}")
                continue

        logger.info(f"Total articles retrieved: {len(all_articles)}")
        return all_articles

    def _get_all_articles_for_law(self, law_num: str) -> list[ArticleWithSummary]:
        """指定した法令番号の全条文をS3 Vectorsから取得する"""
        articles = []
        next_token = None

        while True:
            kwargs = {
                "vectorBucketArn": self.vector_bucket_arn,
                "indexName": self.vector_index_name,
                "returnMetadata": True,
            }

            # メタデータフィルタで法令番号を指定
            kwargs["filter"] = {"conditions": [{"key": "law_num", "value": law_num, "operator": "eq"}]}

            if next_token:
                kwargs["nextToken"] = next_token

            try:
                response = self.s3vectors_client.list_vectors(**kwargs)
            except Exception as e:
                logger.error(f"list_vectors failed for law_num={law_num}: {e}")
                break

            for vector in response.get("vectors", []):
                metadata = vector.get("metadata", {})
                content = metadata.get("content", "")
                law_title = metadata.get("law_title", "")

                # 10万文字超の法令はsummaryのみ
                is_large = len(content) > 100000 if content else False

                article = ArticleWithSummary(
                    law_num=law_num,
                    law_id=metadata.get("law_id", ""),
                    law_title=law_title,
                    unique_anchor=metadata.get("unique_anchor", ""),
                    article_summary=metadata.get("article_summary", ""),
                    content=metadata.get("article_summary", "") if is_large else content,
                    is_summary_only=is_large,
                )
                articles.append(article)

            next_token = response.get("nextToken")
            if not next_token:
                break

        return articles

    def get_full_articles(
        self, law_nums: list[str], unique_anchors: list[str]
    ) -> list[FullArticle]:
        """法令番号とアンカーから条文全文を取得する"""
        if not unique_anchors or not law_nums:
            return []

        full_articles = []

        for unique_anchor in unique_anchors:
            try:
                # ベクトルキーで直接取得
                response = self.s3vectors_client.get_vectors(
                    vectorBucketArn=self.vector_bucket_arn,
                    indexName=self.vector_index_name,
                    keys=[unique_anchor],
                    returnMetadata=True,
                )

                for vector in response.get("vectors", []):
                    metadata = vector.get("metadata", {})
                    law_id = metadata.get("law_id", "")
                    law_title = metadata.get("law_title", "")
                    article_summary = metadata.get("article_summary", "")
                    content = metadata.get("content", "")
                    anchor = metadata.get("anchor")

                    url = f"https://laws.e-gov.go.jp/law/{law_id.split('_')[0]}"
                    if anchor:
                        url = f"https://laws.e-gov.go.jp/law/{law_id}#{anchor}"

                    full_article = FullArticle(
                        law_id=law_id,
                        title=f"{law_title} {article_summary}".strip(),
                        content=content if content else "内容が取得できませんでした。",
                        unique_anchor=unique_anchor,
                        anchor=anchor,
                        url=url,
                    )
                    full_articles.append(full_article)

            except Exception as e:
                logger.warning(f"Failed to get vector for key={unique_anchor}: {e}")
                continue

        return full_articles
