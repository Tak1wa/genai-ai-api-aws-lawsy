variable "aws_region" {
  description = "AWS リージョン"
  type        = string
  default     = "ap-northeast-1"
}

variable "project_name" {
  description = "プロジェクト名（リソース名のプレフィックス）"
  type        = string
  default     = "lawsy-aws"
}


variable "vector_bucket_arn" {
  description = "S3 Vectors バケットの ARN"
  type        = string
}

variable "vector_index_name" {
  description = "S3 Vectors インデックス名"
  type        = string
  default     = "laws-index"
}

variable "model_id" {
  description = "Bedrock モデル ID (Claude)"
  type        = string
  default     = "anthropic.claude-sonnet-4-20250514-v1:0"
}

variable "embedding_model_id" {
  description = "Bedrock Embedding モデル ID"
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "allowed_ip_addresses" {
  description = "API アクセスを許可する IP アドレスのリスト (CIDR)"
  type        = list(string)
  default     = []
}

variable "data_bucket_name" {
  description = "条文データ格納用 S3 バケット名"
  type        = string
}
