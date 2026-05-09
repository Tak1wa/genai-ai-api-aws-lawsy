"""Amazon S3 Vectors + S3 を利用した法令検索の実装

法令名のベクトル検索は S3 Vectors で行い、
条文データの取得は通常の S3 から行う。
"""

import hashlib
import json
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
# S3 Vectors + S3 リトリーバー
# ---------------------------


class S3VectorsRetriever:
    """Amazon S3 Vectors + S3 を使用した法令検索"""

    def __init__(
        self,
        vector_bucket_arn: str,
        vector_index_name: str,
        bedrock_client: BedrockClient,
        data_bucket_name: str = "lawsy-aws-data-PLACEHOLDER",
    ):
        self.vector_bucket_arn = vector_bucket_arn
        # vector_bucket_arn から bucket name を抽出
        # arn:aws:s3vectors:region:account:bucket/name -> name
        self.vector_bucket_name = vector_bucket_arn.split("/")[-1]
        self.vector_index_name = vector_index_name
        self.bedrock_client = bedrock_client
        self.data_bucket_name = data_bucket_name
        self.s3vectors_client = boto3.client(
            "s3vectors", region_name=bedrock_client.config.region
        )
        self.s3_client = boto3.client(
            "s3", region_name=bedrock_client.config.region
        )

        # 法令マッピングをキャッシュ
        self._law_mapping = None

        logger.info(
            f"S3VectorsRetriever initialized: vector_bucket={self.vector_bucket_name}, "
            f"data_bucket={data_bucket_name}, index={vector_index_name}"
        )

    def _get_law_mapping(self) -> dict:
        """法令番号→S3キーのマッピングを取得（キャッシュ付き）"""
        if self._law_mapping is not None:
            return self._law_mapping

        try:
            response = self.s3_client.get_object(
                Bucket=self.data_bucket_name, Key="law_mapping.json"
            )
            self._law_mapping = json.loads(response["Body"].read().decode("utf-8"))
            logger.info(f"Law mapping loaded: {len(self._law_mapping)} entries")
        except Exception as e:
            logger.error(f"Failed to load law mapping: {e}")
            self._law_mapping = {}

        return self._law_mapping

    def get_articles_by_nearest_law(
        self, law_names: list[str]
    ) -> list[ArticleWithSummary]:
        """法令名候補からベクトル検索で最近傍の法令を特定し、S3から条文を取得する"""
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
                    vectorBucketName=self.vector_bucket_name,
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

        # マッチした法令番号の条文をS3から取得
        for law_num in matched_law_nums:
            try:
                articles = self._get_articles_from_s3(law_num)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Failed to get articles for law_num={law_num}: {e}")
                continue

        logger.info(f"Total articles retrieved: {len(all_articles)}")
        return all_articles

    def _get_articles_from_s3(self, law_num: str) -> list[ArticleWithSummary]:
        """S3から指定法令番号の全条文を取得する"""
        mapping = self._get_law_mapping()
        s3_key = mapping.get(law_num)

        if not s3_key:
            # マッピングにない場合はハッシュで直接試行
            key_hash = hashlib.md5(law_num.encode()).hexdigest()[:8]
            s3_key = f"articles/{key_hash}.json"

        try:
            response = self.s3_client.get_object(
                Bucket=self.data_bucket_name, Key=s3_key
            )
            articles_data = json.loads(response["Body"].read().decode("utf-8"))
        except Exception as e:
            logger.error(f"S3 get failed for {s3_key}: {e}")
            return []

        articles = []
        for row in articles_data:
            content = row.get("content", "")
            # 10万文字超の法令はsummaryのみ
            total_chars = sum(
                len(r.get("content", "")) for r in articles_data
            )
            is_large = total_chars > 100000

            article = ArticleWithSummary(
                law_num=row["law_num"],
                law_id=row["law_id"],
                law_title=row["law_title"],
                unique_anchor=row["unique_anchor"],
                article_summary=row.get("article_summary", ""),
                content=row.get("article_summary", "") if is_large else content,
                is_summary_only=is_large,
            )
            articles.append(article)

        return articles

    def get_full_articles(
        self, law_nums: list[str], unique_anchors: list[str]
    ) -> list[FullArticle]:
        """法令番号とアンカーから条文全文を取得する"""
        if not unique_anchors or not law_nums:
            return []

        full_articles = []
        anchor_set = set(unique_anchors)

        for law_num in law_nums:
            try:
                mapping = self._get_law_mapping()
                s3_key = mapping.get(law_num)
                if not s3_key:
                    key_hash = hashlib.md5(law_num.encode()).hexdigest()[:8]
                    s3_key = f"articles/{key_hash}.json"

                response = self.s3_client.get_object(
                    Bucket=self.data_bucket_name, Key=s3_key
                )
                articles_data = json.loads(
                    response["Body"].read().decode("utf-8")
                )

                for row in articles_data:
                    if row["unique_anchor"] in anchor_set:
                        law_id = row["law_id"]
                        anchor = row.get("anchor")
                        url = f"https://laws.e-gov.go.jp/law/{law_id.split('_')[0]}"
                        if anchor:
                            url = f"https://laws.e-gov.go.jp/law/{law_id}#{anchor}"

                        full_article = FullArticle(
                            law_id=law_id,
                            title=f"{row['law_title']} {row.get('article_summary', '')}".strip(),
                            content=row.get("content", ""),
                            unique_anchor=row["unique_anchor"],
                            anchor=anchor,
                            url=url,
                        )
                        full_articles.append(full_article)

            except Exception as e:
                logger.error(f"Failed to get full articles for {law_num}: {e}")
                continue

        return full_articles
