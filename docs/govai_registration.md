# 源内への登録手順

## ExApp 登録パラメータ

源内 Web のチーム管理画面から以下の設定で ExApp を登録してください。

### 基本設定

| 項目 | 値 |
|---|---|
| 名前 | 法令レポート生成（AWS版） |
| APIエンドポイントのURL | `https://<api-id>.execute-api.ap-northeast-1.amazonaws.com/prod/invoke` |
| APIキー | Terraform output の `api_key_value` |

### APIリクエストのデータ形式(JSON)

```json
{"input_text": {"type": "textarea", "title": "法令に関する質問", "desc": "法令に関する質問を入力してください。例：「デジタル社会形成基本法における『デジタル社会』の定義を教えてください」", "required": true}}
```

## API エンドポイントの確認方法

```bash
cd envs/dev
terraform output api_endpoint
terraform output -raw api_key_value
```

## IP制限の設定

源内 Web の NAT Gateway Elastic IP を `allowed_ip_addresses` に追加してください。

```hcl
# terraform.tfvars
allowed_ip_addresses = [
  "xxx.xxx.xxx.xxx/32",  # 源内 Web の NAT Gateway EIP
]
```

## 動作確認

```bash
URL=$(terraform output -raw api_endpoint)
API_KEY=$(terraform output -raw api_key_value)

curl -X POST \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{"inputs": {"input_text": "デジタル社会形成基本法における「デジタル社会」の定義を教えてください"}}' \
  "${URL}"
```

## レスポンス形式

```json
{
  "outputs": "# レポートタイトル\n\n**リード文...**\n\n## セクション...",
  "usageMetadata": [
    {
      "modelVersion": "anthropic.claude-sonnet-4-20250514-v1:0",
      "requestCount": 3,
      "tokens": {
        "inputTokens": 15000,
        "outputTokens": 3000,
        "totalTokens": 18000
      },
      "estimatedCostInfo": {
        "estimatedCost": 0.09,
        "currency": "USD"
      }
    }
  ]
}
```

源内 Web は `outputs` フィールドの Markdown テキストを表示します。
