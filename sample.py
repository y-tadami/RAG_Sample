import os
import streamlit as st
import pdfplumber
import nltk
import faiss
import numpy as np
import tiktoken
from rank_bm25 import BM25Okapi
from openai import AzureOpenAI
from typing import List, Tuple
from janome.tokenizer import Tokenizer
from dotenv import load_dotenv

load_dotenv()

# 定数の定義
GPT_MODEL = "gpt-4o"  # Azure OpenAIで使用している実際のベースモデル名
AZURE_DEPLOYMENT_NAME = os.getenv('AZURE_DEPLOYMENT_NAME')  # Azure OpenAIのデプロイメント名
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL')  # Azure OpenAIの埋め込みモデル名
MAX_TOKENS = 4000
TOP_K = 10
RRF_K = 60

# 必要なツールの初期化
nltk.download('punkt', quiet=True)
enc = tiktoken.get_encoding("cl100k_base")  # GPT-4用のエンコーディング

client = AzureOpenAI(
    api_key = os.getenv('AZURE_API_KEY'),
    api_version = os.getenv('AZURE_API_VERSION'),
    azure_endpoint = os.getenv('AZURE_ENDPOINT'),
    max_retries = 3
)

def count_tokens(text: str) -> int:
    """
    与えられたテキストのトークン数をカウントする関数
    
    この関数は、GPT-4モデル用のエンコーダーを使用して、
    入力テキストのトークン数を計算する。これは、APIリクエストの
    トークン制限を管理するために使用。

    引数:
        text (str): カウントするテキスト

    返り値:
        int: テキスト内のトークン数
    """
    return len(enc.encode(text))

def summarize_text(text: str) -> str:
    """
    GPT-4を使用してテキストを要約する関数
    
    この関数は、Azure OpenAI APIを使用して与えられたテキストの要約を生成する。
    テキストが長すぎる場合は切り詰め、エラーが発生した場合はエラーメッセージを返す。

    引数:
        text (str): 要約するテキスト

    返り値:
        str: 生成された要約またはエラーメッセージ
    """
    if count_tokens(text) > MAX_TOKENS:
        text = text[:MAX_TOKENS]
    
    try:
        response = client.chat.completions.create(
            model=AZURE_DEPLOYMENT_NAME,
            messages=[
                {"role": "system", "content": "あなたは文章を要約する日本語アシスタントです。"},
                {"role": "user", "content": f"以下の文章を日本語で1文に要約してください：\n\n{text}"}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"要約中にエラーが発生しました: {str(e)}")
        return "要約を生成できませんでした。"

def create_embedding(text: str) -> np.ndarray:
    """
    テキストの埋め込みを生成する関数
    
    この関数は、Azure OpenAI APIを使用してテキストの埋め込みベクトルを生成する。

    引数:
        text (str): 埋め込みを生成するテキスト

    返り値:
        np.ndarray: 生成された埋め込みベクトル
    """
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return np.array(response.data[0].embedding, dtype=np.float32)

def vector_search(query: str, index: faiss.IndexFlatIP, embeddings: np.ndarray, k: int = TOP_K) -> Tuple[List[float], List[int], List[int]]:
    """
    ベクトル検索を実行する関数
    
    この関数は、クエリと文書のコサイン類似度に基づいて検索を行う。
    クエリをベクトル化し、FAISSインデックスを使用して最も類似度の高い文書を見つける。

    引数:
        query (str): 検索クエリ
        index (faiss.IndexFlatIP): FAISSインデックス
        embeddings (np.ndarray): 文書の埋め込み
        k (int): 返す結果の数

    返り値:
        Tuple[List[float], List[int], List[int]]: 
            正規化された類似度スコア、文書インデックス、ランクのリスト
    """
    query_embedding = create_embedding(query).reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    similarities, indices = index.search(query_embedding, k)
    
    valid_results = [(sim, idx) for sim, idx in zip(similarities[0], indices[0]) if idx != -1]
    if not valid_results:
        return [], [], []
    
    valid_similarities, valid_indices = zip(*valid_results)
    normalized_similarities = [(sim + 1) / 2 for sim in valid_similarities]
    ranks = list(range(1, len(valid_indices) + 1))

    return list(normalized_similarities), list(valid_indices), ranks

def tokenize(text: str) -> List[str]:
    """
    日本語テキストをトークン化する関数
    
    この関数は、Janomeトークナイザーを使用して日本語テキストをトークン化する。
    助詞と助動詞を除外し、意味のある単語のみの抽出を行う。

    引数:
        text (str): トークン化する日本語テキスト

    返り値:
        List[str]: トークン化された単語のリスト
    """
    tokenizer = Tokenizer()
    return [token.surface for token in tokenizer.tokenize(text) if token.part_of_speech.split(',')[0] not in ['助詞', '助動詞']]

def keyword_search(query: str, documents: List[str], k: int = TOP_K) -> Tuple[List[float], List[int], List[int]]:
    """
    キーワード検索を実行する関数
    
    この関数は、BM25アルゴリズムを使用してキーワード検索を行う。
    クエリと文書をトークン化し、最も関連性の高い文書を見つける。

    引数:
        query (str): 検索クエリ
        documents (List[str]): 検索対象の文書リスト
        k (int): 返す結果の数

    返り値:
        Tuple[List[float], List[int], List[int]]: 
            正規化されたスコア、文書インデックス、ランクのリスト
    """
    tokenized_corpus = [tokenize(doc) for doc in documents]
    tokenized_query = tokenize(query)
    
    bm25 = BM25Okapi(tokenized_corpus)
    doc_scores = bm25.get_scores(tokenized_query)
    
    if np.all(doc_scores == 0):
        return [], [], []
    
    top_k_indices = doc_scores.argsort()[-k:][::-1]
    top_k_scores = doc_scores[top_k_indices]
    
    max_score = np.max(top_k_scores)
    normalized_scores = top_k_scores / max_score if max_score > 0 else np.ones_like(top_k_scores)
    
    ranks = list(range(1, len(top_k_indices) + 1))
    
    return list(normalized_scores), list(top_k_indices), ranks

def rrf_score(rank: int, k: int = RRF_K) -> float:
    """
    Reciprocal Rank Fusion (RRF) スコアを計算する関数
    
    この関数は、与えられたランクに基づいてRRFスコアの計算を行う。
    RRFは複数の検索結果をマージする際に使用される手法。

    引数:
        rank (int): 文書のランク
        k (int): RRF計算のためのパラメータ（デフォルト: 60）

    返り値:
        float: 計算されたRRFスコア
    """
    return 1 / (k + rank)

def hybrid_search_rrf(query: str, index: faiss.IndexFlatIP, embeddings: np.ndarray, documents: List[str], k: int = TOP_K) -> Tuple[List[float], List[int], List[int], List[int]]:
    """
    ハイブリッド検索を実行する関数
    
    この関数は、ベクトル検索とキーワード検索の結果を組み合わせて
    Reciprocal Rank Fusion (RRF)スコアを計算し、最終的な検索結果を生成する。

    引数:
        query (str): 検索クエリ
        index (faiss.IndexFlatIP): FAISSインデックス
        embeddings (np.ndarray): 文書の埋め込み
        documents (List[str]): 検索対象の文書リスト
        k (int): 返す結果の数

    返り値:
        Tuple[List[float], List[int], List[int], List[int]]: 
            正規化されたスコア、文書インデックス、ベクトル検索ランク、キーワード検索ランクのリスト
    """
    vec_similarities, vec_indices, _ = vector_search(query, index, embeddings, k)
    key_similarities, key_indices, _ = keyword_search(query, documents, k)
    
    vec_ranks = {idx: rank for rank, idx in enumerate(vec_indices)}
    key_ranks = {idx: rank for rank, idx in enumerate(key_indices)}
    all_indices = set(vec_indices) | set(key_indices)
    
    rrf_scores = {idx: rrf_score(vec_ranks.get(idx, len(documents))) + rrf_score(key_ranks.get(idx, len(documents))) for idx in all_indices}
    
    sorted_indices = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]
    sorted_scores = [rrf_scores[idx] for idx in sorted_indices]
    
    max_score = max(sorted_scores) if sorted_scores else 1
    normalized_scores = [score / max_score for score in sorted_scores]
    
    vec_ranks_result = [vec_ranks.get(idx, len(documents)) + 1 for idx in sorted_indices]
    key_ranks_result = [key_ranks.get(idx, len(documents)) + 1 for idx in sorted_indices]
    
    return normalized_scores, sorted_indices, vec_ranks_result, key_ranks_result

def extract_text_from_pdf(uploaded_file) -> str:
    """
    アップロードされたPDFファイルからテキストを抽出する関数
    
    この関数は、pdfplumberライブラリを使用してPDFファイルからテキストを抽出する。
    各ページのテキストを抽出し、それらを結合して1つの文字列として返す。

    引数:
        uploaded_file: Streamlitでアップロードされたファイルオブジェクト

    返り値:
        str: 抽出されたテキスト

    例外処理:
        ValueError: PDFからテキストを抽出できなかった場合
    """
    with pdfplumber.open(uploaded_file) as pdf:
        text = "\n\n".join(page.extract_text(x_tolerance=3, y_tolerance=3) or "" for page in pdf.pages)
    if not text.strip():
        raise ValueError("PDFからテキストを抽出できませんでした。")
    return text

def chunk_text(text: str) -> List[str]:
    """
    テキストをチャンクに分割する関数
    
    この関数は、入力テキストを段落または文単位で分割する。
    段落が少ない場合（10未満）は、文単位で分割を行う。

    引数:
        text (str): 分割するテキスト

    返り値:
        List[str]: 分割されたテキストチャンクのリスト
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if len(paragraphs) < 10:
        sentences = nltk.sent_tokenize(text)
        return [sent for sent in sentences if len(sent.split()) > 5]
    return paragraphs

def create_embeddings_and_index(chunked_result: List[str]) -> Tuple[np.ndarray, faiss.IndexFlatIP]:
    """
    テキストチャンクの埋め込みを生成し、検索インデックスを作成する関数
    
    この関数は、テキストチャンクをベクトル化し、FAISSインデックスを作成する。
    これにより、効率的なベクトル検索や、キーワード検索を行うことができる。

    引数:
        chunked_result (List[str]): テキストチャンクのリスト

    返り値:
        Tuple[np.ndarray, faiss.IndexFlatIP]: 
            生成された埋め込みとFAISSインデックス
    """
    embeddings = np.array([create_embedding(chunk) for chunk in chunked_result])
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return embeddings, index

def process_pdf(uploaded_file) -> Tuple[List[str], np.ndarray, faiss.IndexFlatIP]:
    """
    PDFファイルを処理し、テキストチャンク、埋め込み、検索インデックスを生成する関数
    
    この関数は、PDFファイルからテキストを抽出し、チャンクに分割し、
    埋め込みとインデックスを生成する一連の処理を行う。

    引数:
        uploaded_file: Streamlitでアップロードされたファイルオブジェクト

    返り値:
        Tuple[List[str], np.ndarray, faiss.IndexFlatIP]: 
            テキストチャンク、埋め込み、FAISSインデックス
    """
    text = extract_text_from_pdf(uploaded_file)
    chunked_result = chunk_text(text)
    embeddings, index = create_embeddings_and_index(chunked_result)
    return chunked_result, embeddings, index

def display_search_results(similarities: List[float], indices: List[int], chunked_result: List[str], vec_ranks: List[int] = None, key_ranks: List[int] = None):
    """
    検索結果を表示し、各結果の要約を生成する関数
    
    この関数は、検索結果をStreamlitインターフェース上に表示する。
    各結果に対して要約を生成し、スコアやランク情報とともに表示する。
    また、上位の結果を用いて全体の要約も生成する。

    引数:
        similarities (List[float]): 各結果の類似度スコア
        indices (List[int]): 結果の文書インデックス
        chunked_result (List[str]): 元のテキストチャンク
        vec_ranks (List[int], optional): ベクトル検索でのランク
        key_ranks (List[int], optional): キーワード検索でのランク
    """
    st.subheader(f"Top {min(TOP_K, len(similarities))} results:")
    summaries = []
    for i, (similarity, idx) in enumerate(zip(similarities, indices)):
        if i >= TOP_K:
            break
        original_text = chunked_result[idx]
        with st.spinner(f"結果 {i+1} を要約中..."):
            summary = summarize_text(original_text)
        summaries.append(summary)
        st.write(f"{i+1}. スコア: {similarity:.4f}")
        if vec_ranks and key_ranks:
            st.write(f"   ベクトル検索順位: {vec_ranks[i]}, キーワード検索順位: {key_ranks[i]}")
        st.write(f"   検索結果: {summary}")
        st.write("---")

    if summaries:
        with st.spinner("全体の要約を生成中..."):
            overall_summary_text = "以下の要約をさらに1文で要約してください：\n\n" + "\n".join(summaries[:5])
            overall_summary = summarize_text(overall_summary_text)
        st.subheader("検索結果の全体要約:")
        st.write(overall_summary)

def main():
    """
    アプリケーションの主要な処理を行う関数
    
    この関数は、Streamlitアプリケーションの起動時に実行され、
    ユーザーインターフェースの構築や主要な機能の制御を行う。
    PDFのアップロード、テキスト処理、検索機能の選択、
    クエリの入力、結果の表示など、アプリケーションの
    中心となる処理をこの関数内で管理している。
    """
    st.title("PDF文書検索アプリケーション")

    uploaded_file = st.file_uploader("PDFファイルを選択してください：", type="pdf")

    if uploaded_file is not None:
        try:
            with st.spinner("PDFを処理中..."):
                chunked_result, embeddings, index = process_pdf(uploaded_file)
            
            st.success(f"PDFの処理が完了しました。{len(chunked_result)}個のチャンクに分割されました。")
            
            search_method = st.radio("検索方法を選択してください：", ("ベクトル検索", "キーワード検索", "ハイブリッド検索"))
            query = st.text_input("検索ワードを入力してください：")

            if query:
                with st.spinner("検索中..."):
                    if search_method == "ベクトル検索":
                        similarities, indices, ranks = vector_search(query, index, embeddings)
                        st.info("スコアはコサイン類似度を使用して計算されています。")
                        display_search_results(similarities, indices, chunked_result, ranks, None)
                    elif search_method == "キーワード検索":
                        similarities, indices, ranks = keyword_search(query, chunked_result)
                        st.info("スコアはBM25アルゴリズムを使用して計算されています。")
                        display_search_results(similarities, indices, chunked_result, None, ranks)
                    elif search_method == "ハイブリッド検索":
                        similarities, indices, vec_ranks, key_ranks = hybrid_search_rrf(query, index, embeddings, chunked_result)
                        st.info("スコアはReciprocal Rank Fusion (RRF)を使用して計算されています。")
                        display_search_results(similarities, indices, chunked_result, vec_ranks, key_ranks)
                    else:
                        st.error("無効な検索方法が選択されました。")
                        return
                    
                    if not similarities or not indices:
                        st.warning("検索結果が見つかりませんでした。")
                        return

        except Exception as e:
            st.error(f"PDFの処理中にエラーが発生しました: {str(e)}")
            st.exception(e)
    else:
        st.warning("PDFファイルがアップロードされていません。")

if __name__ == "__main__":
    main()