# Lawsy-Custom-S3Vectors (AWS版)

法令に関する質問を受け取り、AIが意図を解析して最適化されたレポートを動的に生成するサーバーレス API システムです。
Amazon S3 Vectors のベクトル検索と Amazon Bedrock (Claude) を組み合わせ、日本の法令データ（e-Gov）を対象とした検索・回答生成を行います。

本プロジェクトは [digital-go-jp/genai-ai-api](https://github.com/digital-go-jp/genai-ai-api) の Google Cloud 版 Lawsy-Custom-BQ を AWS に移植したものです。

## 技術スタック

- バックエンド: Python 3.12 (AWS Lambda)
- AI モデル: Amazon Bedrock (Claude 3.5 Sonnet v2)
- ベクトル検索: Amazon S3 Vectors
- Embedding: Amazon Bedrock (Amazon Titan Text Embeddings V2)
- Web検索: Amazon Bedrock (Tavily Web Search Tool)
- インフラ: AWS (API Gateway, Lambda, S3 Vectors)
- IaC: Terraform

## アーキテクチャ

```
源内 Web (Lambda/VPC) 
    → NAT Gateway 
    → Amazon API Gateway (API Key認証 + IP制限)
    → AWS Lambda (Python 3.12)
        → Amazon Bedrock (Claude 3.5 Sonnet v2) - レポート生成
        → Amazon Bedrock (Titan Embeddings V2) - ベクトル埋め込み
        → Amazon S3 Vectors - 法令ベクトル検索
        → Amazon Bedrock Web Search - 法令名推定
```

## 機能

1. **法令検索**: S3 Vectors のベクトル検索で関連法令を高速検索
2. **レポート生成**: クエリの意図を判断して6パターンから最適な構造を選び、包括的なレポートを1回の AI 呼び出しで生成
3. **出典管理**: 実際に引用された参考情報のみを整理して表示

## 処理の流れ

### Step 1: 法令名推定・検索
- Bedrock Web Search で関連法令名を推定（3段階フォールバック付き）
- 施行令・施行規則を補完して検索対象を拡張
- S3 Vectors ベクトル検索で条文を取得（並列で Web 検索も実行）

### Step 2: 条文選択と完全レポート生成
- AI が関連条文を選択し、全文を取得
- クエリの意図から6パターンの最適な構造を自動選択
- 検索結果を統合し Markdown レポートを1回で生成

### Step 3: 出典フィルタリングと後処理
- 実際に引用された参考情報のみを抽出
- クリック可能なリンク形式で整理
- Mermaid 図表の sanitize、引用リンクの外部リンク化

## デプロイ方法

### 前提条件

- AWS CLI v2 がインストール・設定済み
- Terraform >= 1.5
- Python 3.12
- 以下の AWS サービスが利用可能:
  - Amazon Bedrock (Claude 3.5 Sonnet v2, Titan Text Embeddings V2 のモデルアクセスが有効)
  - Amazon S3 Vectors
  - AWS Lambda
  - Amazon API Gateway

### データ準備

1. e-Gov から法令 XML を一括ダウンロード
2. 前処理パイプラインを実行:

```bash
pip install -r preprocess/requirements.txt

python preprocess/run_entire_pipeline.py \
  --region ap-northeast-1 \
  --vector-bucket-name your-vector-bucket \
  --vector-index-name laws-index \
  --data-bucket-name your-data-bucket \
  --xml-dir ./xml_files \
  --date-tag 20250509
```

### Terraform デプロイ

```bash
cp -r envs/sample envs/dev
# envs/dev/terraform.tfvars を編集

cd envs/dev
terraform init
terraform plan
terraform apply
```

### デプロイ後の確認

```bash
URL=$(terraform output -raw api_endpoint)
API_KEY=$(terraform output -raw api_key_value)

curl -X POST \
  -H "Content-Type: application/json" \
  -H "x-api-key: ${API_KEY}" \
  -d '{"inputs": {"input_text": "デジタル社会形成基本法における「デジタル社会」の定義を教えてください"}}' \
  "${URL}/invoke"
```

## 源内への登録

デプロイ後、源内 Web のチーム管理画面から ExApp を登録:

- 名前: 法令レポート生成
- APIエンドポイントのURL: `https://xxxxxxxx.execute-api.ap-northeast-1.amazonaws.com/prod/invoke`
- APIキー: Terraform output の `api_key_value`
- APIリクエストのデータ形式(JSON):

```json
{"input_text": {"type": "textarea", "title": "法令に関する質問", "desc": "法令に関する質問を入力してください。例：「デジタル社会形成基本法における『デジタル社会』の定義を教えてください」", "required": true}}
```

## ランニングコスト試算

| サービス | 料金体系 | 月額目安（検証用途） |
|---|---|---|
| Lambda | リクエスト + 実行時間従量課金 | ほぼ無料（無料枠内） |
| API Gateway | $1/100万コール | $0（検証レベル） |
| S3 Vectors | ストレージ + クエリ従量課金 | $1〜3 |
| Bedrock (Claude) | 入出力トークン従量課金 | $1〜10 |
| Bedrock (Titan Embeddings) | トークン従量課金 | $0.1未満 |
| S3 (データ保管) | $0.025/GB | $0.1未満 |

検証用途では月額 **$3〜15** 程度で運用可能です。

## ライセンス

MIT License
