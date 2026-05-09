variable "aws_region" {
  description = "AWS リージョン"
  type        = string
}

variable "project_name" {
  description = "プロジェクト名"
  type        = string
}

variable "vector_bucket_arn" {
  description = "S3 Vectors バケットの ARN"
  type        = string
}

variable "vector_index_name" {
  description = "S3 Vectors インデックス名"
  type        = string
}

variable "model_id" {
  description = "Bedrock モデル ID"
  type        = string
}

variable "embedding_model_id" {
  description = "Bedrock Embedding モデル ID"
  type        = string
}

variable "allowed_ip_addresses" {
  description = "API アクセスを許可する IP アドレスのリスト"
  type        = list(string)
  default     = []
}

variable "data_bucket_name" {
  description = "条文データ格納用 S3 バケット名"
  type        = string
}
