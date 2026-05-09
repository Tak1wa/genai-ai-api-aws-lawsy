output "api_endpoint" {
  description = "API Gateway エンドポイント URL"
  value       = "${aws_api_gateway_stage.prod.invoke_url}/invoke"
}

output "api_key_value" {
  description = "API Key の値"
  value       = aws_api_gateway_api_key.lawsy.value
  sensitive   = true
}

output "lambda_function_name" {
  description = "Lambda 関数名"
  value       = aws_lambda_function.lawsy.function_name
}

output "api_gateway_id" {
  description = "API Gateway ID"
  value       = aws_api_gateway_rest_api.lawsy.id
}

output "lambda_function_url" {
  description = "Lambda Function URL (タイムアウト制限なし)"
  value       = aws_lambda_function_url.lawsy.function_url
}
