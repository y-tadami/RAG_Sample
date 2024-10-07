# PDF文書検索アプリケーション

### 1. アプリケーションの目的
このアプリケーションは、PDFファイルをアップロードし、その内容を検索可能にするツールです。
ベクトル検索、キーワード検索、ハイブリッド検索の3つの検索方法を提供し、
検索結果の要約も生成します。

### 2. 主な機能
-  PDFファイルのアップロードと処理
- テキストの抽出とチャンク分割
- ベクトル検索（コサイン類似度ベース）
- キーワード検索（BM25アルゴリズム使用）
- ハイブリッド検索（ベクトル検索とキーワード検索の組み合わせ）
- 検索結果の要約生成
- 全体の要約生成

### 3. 実行環境の想定条件
- Python 3.8以上
- pip（Pythonパッケージマネージャー）
- 仮想環境（venvまたはAnaconda）

### 4. インストール手順
1. リポジトリをクローンする：

   git clone https://github.com/yourusername/pdf-search-app.git
   
   cd pdf-search-app

3. 仮想環境を作成し、アクティベートする：
   python -m venv venv
   
   source venv/bin/activate  # Linuxの場合
   
   venv\Scripts\activate  # Windowsの場合

5. 必要なパッケージをインストールする：
   pip install -r requirements.txt

### 5. 使用方法
1. アプリケーションを起動する：
   streamlit run app.py

2. ブラウザで表示されたローカルURLにアクセスする（通常 http://localhost:8501）

3. PDFファイルをアップロードし、検索方法（ベクトル、キーワード、ハイブリッド）をラジオボタンによって選択して検索クエリを入力する

4. 検索結果と要約を確認する

### 6. 環境変数の設定
以下の環境変数を .env ファイルに設定してください：

AZURE_API_KEY=your_azure_api_key
AZURE_ENDPOINT=your_azure_endpoint
AZURE_API_VERSION=your_azure_api_version
AZURE_DEPLOYMENT_NAME=your_deployment_name
EMBEDDING_MODEL=your_embedding_model_name

### 7. 注意事項
- 大きなPDFファイルの処理には時間がかかる場合があります
- API使用量に注意してください（Azure OpenAIの利用制限に注意）
- 検索結果の精度は入力PDFの品質に依存します

### 8.トラブルシューティング
- faiss-cpuのインストールに問題がある場合は、システムに応じた追加の設定が必要な場合があります
- PDFの抽出に失敗する場合は、PDFファイルが破損していないか確認してください
