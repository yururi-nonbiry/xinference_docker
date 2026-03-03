# **Xinference環境におけるruri-v3-310mおよびruri-v3-reranker-largeの統合と最適化に関する包括的研究報告書**

## **日本語セマンティック検索の変遷と高精度埋め込みモデルの必要性**

日本語の自然言語処理（NLP）は、その言語的特性から長らく独自の進化を遂げてきた。英語とは異なり単語間の空白（スペース）が存在しない日本語では、分かち書きと呼ばれるプロセスが必須であり、MeCabやSudachiといった形態素解析器がテキスト処理の基盤を支えてきた 1。しかし、大規模言語モデル（LLM）の台頭に伴い、テキストを多次元のベクトル空間に写像する「埋め込み（Embedding）」技術が、従来のキーワードマッチング（BM25等）を補完、あるいは凌駕する手法として注目を集めている 2。

この文脈において、名古屋大学大学院情報学研究科の計算言語学研究室（CL Research Group）が開発した「Ruri」シリーズは、日本語に特化した汎用テキスト埋め込みモデルおよびリランカーとして、現在の最先端（State-of-the-art）の地位を確立している 3。特に最新世代であるRuri-v3は、ModernBERTアーキテクチャを採用することで、従来のモデルが抱えていた最大入力トークン数の制限を劇的に改善し、長文ドキュメントのセマンティック検索において圧倒的な精度を実現している 3。

本報告書では、これら最新の日本語モデルを、オープンソースのモデル推論プラットフォームである「Xinference（Xorbits Inference）」上で展開し、効率的に運用するための技術的手法、構成定義、および最適化戦略について詳述する。Xinferenceは、LLMだけでなく埋め込みモデルやリランカー、さらには画像・音声モデルまでを一元的に管理できる強力なオーケストレーションツールであり、分散環境でのデプロイメントを容易にする特性を持つ 6。

## **Ruri-v3モデルシリーズの技術的構造と優位性**

Ruri-v3は、従来のBERTやRoBERTaをベースとしたモデルと比較して、いくつかの決定的な技術的進歩を遂げている。これらは、日本語の長文処理や推論の効率化において極めて重要な役割を果たす。

## **ModernBERTアーキテクチャへの移行**

Ruri-v3の最大の変更点は、バックボーンとしてModernBERTを採用したことである 3。ModernBERTは、従来のBERTの設計を現代的に再構築したものであり、以下の技術要素を含んでいる。

1. **FlashAttention-2の統合**: 注意機構（Attention Mechanism）の計算効率を飛躍的に向上させ、メモリ使用量を抑えつつ高速な推論を可能にする 3。  
2. **Rotary Positional Embeddings (RoPE)**: 相対的な位置情報をエンコードするRoPEの採用により、学習時を超える長さのシーケンスに対しても安定した性能を発揮する 3。  
3. **8,192トークンのコンテキスト窓**: Ruri-v1およびv2では最大512トークンに制限されていたが、v3では8,192トークンまで対応可能となった。これにより、契約書や学術論文、技術マニュアルといった長大な日本語文書を分割することなく一度に処理できるようになった 3。

## **トークナイザーの革新と語彙の拡大**

日本語処理において、トークナイゼーションはモデルの性能を左右する。Ruri-v3では、SentencePieceのみに基づいたトークナイザーを採用している 3。これにより、外部の単語分割ツールを必要とせず、日本語特有のBERTトークナイザーに依存していた従来モデルの制約を解消している 3。

| 項目 | Ruri-v1 / v2 | Ruri-v3 (310m / Reranker) |
| :---- | :---- | :---- |
| アーキテクチャ | BERT / RoBERTa | ModernBERT |
| 最大シーケンス長 | 512 トークン | 8,192 トークン |
| 語彙サイズ | 約32,000 | 100,000 |
| トークナイザー | MeCab \+ WordPiece | SentencePiece Only |
| 学習言語 | 日本語主体 | 日本語（強化された語彙） |

3。

語彙サイズが10万に拡大されたことで、一つの単語が細切れのトークンに分解される率が低下し、入力シーケンス長が短縮される。これは推論時の計算コスト削減と、モデルが捉える意味的まとまりの維持に直結する 3。

## **Xinferenceのシステムアーキテクチャとカスタムモデル管理**

Xinferenceは、スケーラブルな推論環境を提供するために、スーパーバイザー（Supervisor）とワーカー（Worker）の役割を分離したクラスタ構造をサポートしている 8。モデルのロード時には、専用の仮想環境（Virtual Environment）を自動生成する機能を備えており、依存関係の競合を回避する設計がなされている 9。

## **モデル・ハブと登録メカニズム**

Xinferenceには多数の組み込み（Built-in）モデルが存在するが、Ruri-v3シリーズは現時点では組み込みリストに含まれていない 11。そのため、ユーザーはXinferenceの「カスタムモデル登録」機能を利用して、これらのモデルを定義する必要がある 11。

カスタムモデルの登録は、主にJSON形式の構成ファイルを用いて行われる。このファイルには、モデルの名称、能力（埋め込み、生成、リランクなど）、ハブID（HuggingFace ID等）、および必要なハイパーパラメータを記述する 12。

## **モデル・タイプ別の機能定義**

Xinferenceでは、モデルの用途に応じて MODEL\_TYPE を指定する必要がある 6。

* **embedding**: テキストを固定次元のベクトルに変換するモデル 6。  
* **rerank**: クエリとドキュメントのペアに対して、関連性スコアを付与するモデル 6。

Ruri-v3-310mは embedding として、ruri-v3-reranker-large（またはruri-v3-reranker-310m）は rerank として登録・運用される。

## **ruri-v3-310m（埋め込みモデル）の導入手順**

ruri-v3-310mは、JMTEBベンチマークにおいて極めて高い平均スコア（77.24）を記録している高性能埋め込みモデルである 3。これをXinferenceで利用するためには、以下の詳細な手順を遵守する必要がある。

## **カスタムモデル構成の定義**

まず、ruri-v3-310mの仕様に基づいたJSONファイルを生成する。このモデルは315Mのパラメータを持ち、出力次元数は768である 3。

JSON

{  
  "model\_name": "ruri-v3-310m",  
  "dimensions": 768,  
  "max\_tokens": 8192,  
  "language": \["ja"\],  
  "model\_id": "cl-nagoya/ruri-v3-310m",  
  "model\_specs": \[  
    {  
      "model\_format": "pytorch",  
      "model\_hub": "huggingface"  
    }  
  \]  
}

この定義において、dimensions と max\_tokens の値はモデルのアーキテクチャと厳密に一致していなければならない 12。これらの値が誤っていると、ベクトルの保存先であるベクトルデータベース（MilvusやQdrant等）との整合性が失われる、あるいは長文入力時にモデルがエラーを返す原因となる。

## **登録と起動のプロセス**

作成したJSONファイルを用いて、CLIから登録を行う。

Bash

xinference register \--model-type embedding \--file ruri\_v3\_310m.json \--persist

\--persist オプションは、スーパーバイザーを再起動した後も登録情報を保持するために重要である 11。登録完了後、以下のコマンドでモデルを起動する。

Bash

xinference launch \--model-name ruri-v3-310m \--model-type embedding

起動が成功すると、一意の model\_uid が返される。デフォルトではモデル名がUIDとなるが、特定のUIDを指定することも可能である 15。

## **ruri-v3-reranker-large（リランカー）の導入手順**

リランカーは、ファーストステージの検索（ベクトル検索等）で得られた候補ドキュメントを、より精緻に再順位付けするために使用される 2。Ruri-v3-reranker-large（ruri-v3-reranker-310m）は、クエリとドキュメントのクロスアテンションを計算するクロスエンコーダー（Cross-Encoder）形式を採用しており、JQaRAベンチマークで77.1という高いスコアを叩き出している 4。

## **リランカー特有の登録設定**

リランカーの登録には、埋め込みモデルとは異なるパラメータが必要となる。特に type フィールドには normal（一般的なクロスエンコーダーを指す）を指定する 13。

JSON

{  
  "model\_name": "ruri-v3-reranker-large",  
  "type": "normal",  
  "language": \["ja"\],  
  "model\_id": "cl-nagoya/ruri-v3-reranker-310m",  
  "max\_tokens": 8192,  
  "model\_specs": \[  
    {  
      "model\_format": "pytorch",  
      "model\_hub": "huggingface"  
    }  
  \]  
}

Ruri-v3のリランカーは、埋め込みモデルと同様に8,192トークンに対応しているため、max\_tokens を適切に設定することで長文ドキュメントのリランク精度を最大化できる 5。

## **リランク処理のメカニズム**

クロスエンコーダーは、数学的に以下の注意機構の計算をクエリ ![][image1] とドキュメント ![][image2] の連結シーケンスに対して行う 18。

![][image3]  
このプロセスにより、クエリの各トークンがドキュメントのどの部分に強く関連しているかを詳細に分析できるが、ドキュメント数が増えると計算コストが飛躍的に増大する 18。そのため、Xinferenceでリランカーを運用する際は、GPUリソースの割り当てに注意が必要である。

## **仮想環境管理と日本語固有の依存関係の解決**

Xinference v2.0以降、モデルごとに独立した仮想環境がデフォルトで有効になっている 9。これは、異なるモデルが必要とするライブラリのバージョン衝突を防ぐための重要な機能である。

## **日本語トークナイザーの依存パッケージ**

Ruri-v3はSentencePieceを使用するため、仮想環境内に sentencepiece ライブラリがインストールされている必要がある 3。また、もし古いRuriモデル（v1, v2）や他の日本語モデル（BERTベース）を使用する場合には、fugashi や unidic-lite といった形態素解析用のパッケージが必須となる場合がある 21。

モデル起動時にこれらのパッケージを明示的に指定することで、実行時の「ModuleNotFoundError」を回避できる 9。

Bash

xinference launch \--model-name ruri-v3-310m \\  
                  \--model-type embedding \\  
                  \--virtual-env-package "sentencepiece" \\  
                  \--virtual-env-package "transformers\>=4.48.0"

## **uvツールの活用と高速化**

Xinferenceは仮想環境の作成に uv ツールを使用しており、これは標準の pip よりも大幅に高速である 9。さらに、環境変数 XINFERENCE\_VIRTUAL\_ENV\_SKIP\_INSTALLED=1 を設定することで、システム側に既にインストールされている巨大なライブラリ（PyTorchやCUDA関連）の再インストールをスキップし、ディスク容量の節約とデプロイ時間の短縮が可能となる 10。

## **Python SDKによるプログラムからの操作と統合**

モデルの登録と起動が完了すれば、プログラム（Python）からこれらのモデルにアクセスできる。XinferenceはRESTfulなAPIを提供しており、専用のクライアントライブラリを通じて操作が可能である 24。

## **埋め込みモデルの呼び出し**

XinferenceEmbeddings クラス（LangChain互換）や直接のクライアントメソッドを使用して、テキストをベクトル化する 8。

Python

from xinference.client import Client

client \= Client("http://localhost:9997")  
model \= client.get\_model("ruri-v3-310m")

texts \= \["こんにちは、世界。", "セマンティック検索のデモです。"\]  
embeddings \= model.create\_embedding(texts)

この呼び出しにより、各テキストに対して768次元の浮動小数点配列が生成される 3。

## **リランカーの呼び出し**

リランカーの場合は、rerank メソッドを使用してクエリとドキュメントリストの関連性を評価する 26。

Python

query \= "埋め込みモデルの使い方を教えてください。"  
documents \=

result \= model.rerank(documents, query)

APIからのレスポンスには、各ドキュメントのインデックスと関連性スコア（Relevance Score）が含まれており、これに基づいて元のリストをソートし直すことで、検索精度の向上を図る 26。

## **精度検証：Ruri-v3のベンチマークパフォーマンスの分析**

Ruri-v3の導入効果を定量的に評価するため、公開されているベンチマークデータを参照する。特に、日本語のエンベディング性能を総合的に評価するJMTEB（Japanese Massive Text Embedding Benchmark）の結果は、モデル選択の強力な根拠となる 3。

## **JMTEBにおける埋め込み性能の比較**

以下の表は、Ruri-v3シリーズと、よく比較対象となる他の日本語対応モデルの性能を示している。

| モデル名 | パラメータ数 | 平均スコア (JMTEB) | 出力次元 |
| :---- | :---- | :---- | :---- |
| **ruri-v3-310m** | 315M | 77.24 | 768 |
| ruri-v3-130m | 132M | 76.55 | 512 |
| ruri-v3-70m | 70M | 75.48 | 384 |
| OpenAI text-embedding-3-small | N/A | 約65-70 | 1536 |

3。

Ruri-v3-310mは、商用APIであるOpenAIのモデルを凌駕する性能をオープンソースで提供しており、オンプレミス環境や機密データの処理において極めて高い有用性を持つことがわかる。

## **JQaRAにおけるリランカー性能の比較**

リランカーの性能評価には、日本語の質問応答と検索に特化したJQaRAが用いられる 4。

| モデル名 | JQaRA スコア | 特徴 |
| :---- | :---- | :---- |
| **ruri-v3-reranker-large** | 77.1 | ModernBERTベース、8k対応 |
| BGE-reranker-v2-m3 | 67.3 | 多言語、512トークン |
| ruri-reranker-base (v2) | 74.3 | BERTベース、512トークン |

4。

ruri-v3-reranker-largeは、既存の最高性能モデルであったBGE-reranker-v2-m3を大きく引き離しており、特に日本語における「意味の取り違え」が少ないことが、クロスエンコーダーの強みを最大限に引き出している結果と言える 4。

## **運用上の最適化戦略：レイヤード・リリーバル（2段階検索）**

実稼働システムにおいて、数百万件のドキュメントすべてをリランカーで処理することは、レイテンシの観点から不可能である 17。そのため、Xinferenceでruri-v3-310mとruri-v3-reranker-largeを組み合わせて使用する「2段階検索（Two-stage Retrieval）」が推奨される 2。

## **ステージ1：高速な候補抽出（埋め込みモデル）**

1. ユーザーのクエリを ruri-v3-310m でベクトル化する 2。  
2. ベクトルデータベース上で近似最近傍探索（ANN）を行い、上位50〜100件程度のドキュメントを抽出する 2。  
3. この段階の処理速度はミリ秒単位であり、大規模データセットでも高速に動作する 2。

## **ステージ2：精密な再順位付け（リランカー）**

1. ステージ1で得られた100件のドキュメントとクエリを ruri-v3-reranker-large に渡す 2。  
2. 各ペアの関連性スコアを詳細に計算し、最も関連性の高いドキュメントを最上位に配置する 2。  
3. このプロセスにより、埋め込みモデル単体では捉えきれなかった微妙な意味的差異やキーワードの一致を修正し、最終的な回答の精度を大幅に向上させる 2。

## **トラブルシューティングと既知の課題**

Xinferenceにおけるモデル運用では、インフラ環境に起因するいくつかの問題が発生する可能性がある。

1. **GPUメモリの不足（OOM）**: 特にModernBERTベースのモデルを8,192トークンフルに使用する場合、アテンション行列のサイズが大きくなるため、VRAM消費量が急増する 17。Xinferenceのモデルメモリ計算機能（Model Memory Calculation）を活用し、適切なサイズのGPUを選択することが重要である 6。  
2. **接続タイムアウト**: リランカーで大量のドキュメント（例：500件以上）を一度に処理しようとすると、推論時間が長くなり、クライアント側（RAGFlowやLangChain等）でタイムアウトが発生することがある 17。バッチサイズを調整するか、Xinferenceのワーカー数を増やして並列処理能力を高めることで解決できる。  
3. **登録の永続性**: 前述の通り、--persist フラグを忘れると、コンテナの再起動時にカスタムモデルの定義が消失する 11。運用の自動化にあたっては、起動スクリプト内でモデルの登録状態を確認し、必要に応じて再登録するロジックを組み込むのが一般的である。

## **結論：日本語RAGシステムの次世代標準としてのRuri-v3**

本調査報告が示す通り、Xinference上で ruri-v3-310m と ruri-v3-reranker-large を運用することは、現時点で最高水準の日本語検索基盤を構築するための最適解の一つである。ModernBERTの導入による8,192トークンのサポートは、これまでの512トークンの壁を打破し、複雑で長大な日本語ビジネス文書の処理を可能にした 3。

Xinferenceの提供するモデル管理機能、特に仮想環境による依存関係の分離とREST APIを通じた容易な統合は、これらの高度なモデルをプロダクション環境に導入する際の技術的ハードルを著しく下げている 9。今後、日本語LLMの活用がさらに進む中で、これら高品質な「モデルの目」となる埋め込み技術と、「モデルのフィルター」となるリランク技術の重要性はますます高まっていく。

技術者は、単一のモデルの性能に頼るのではなく、Xinferenceのようなプラットフォームを介して、埋め込みモデルによる広域検索とリランカーによる局所評価を組み合わせた多層的なアーキテクチャを設計することが、真にユーザーに価値を届ける検索体験の実現に繋がると結論付けられる 2。

#### **引用文献**

1. An Experimental Evaluation of Japanese Tokenizers for Sentiment-Based Text Classification \- arXiv, 3月 3, 2026にアクセス、 [https://arxiv.org/pdf/2412.17361](https://arxiv.org/pdf/2412.17361)  
2. Search That Actually Works: A Guide to LLM Rerankers \- DeepInfra, 3月 3, 2026にアクセス、 [https://deepinfra.com/blog/llm-rerankers](https://deepinfra.com/blog/llm-rerankers)  
3. cl-nagoya/ruri-v3-310m \- Hugging Face, 3月 3, 2026にアクセス、 [https://huggingface.co/cl-nagoya/ruri-v3-310m](https://huggingface.co/cl-nagoya/ruri-v3-310m)  
4. cl-nagoya/ruri-reranker-large \- Hugging Face, 3月 3, 2026にアクセス、 [https://huggingface.co/cl-nagoya/ruri-reranker-large](https://huggingface.co/cl-nagoya/ruri-reranker-large)  
5. cl-nagoya/ruri-v3-reranker-310m \- Hugging Face, 3月 3, 2026にアクセス、 [https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m](https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m)  
6. Models \- Xinference \- Read the Docs, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/latest/models/index.html](https://inference.readthedocs.io/en/latest/models/index.html)  
7. llama-index-postprocessor-xinference-rerank \- PyPI, 3月 3, 2026にアクセス、 [https://pypi.org/project/llama-index-postprocessor-xinference-rerank/](https://pypi.org/project/llama-index-postprocessor-xinference-rerank/)  
8. XinferenceEmbeddings | langchain\_community \- LangChain Reference Docs, 3月 3, 2026にアクセス、 [https://reference.langchain.com/python/langchain-community/embeddings/xinference/XinferenceEmbeddings](https://reference.langchain.com/python/langchain-community/embeddings/xinference/XinferenceEmbeddings)  
9. Model Virtual Environments \- Xinference, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/stable/models/virtualenv.html](https://inference.readthedocs.io/en/stable/models/virtualenv.html)  
10. virtualenv.rst.txt \- Xinference, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v1.9.0/\_sources/models/virtualenv.rst.txt](https://inference.readthedocs.io/en/v1.9.0/_sources/models/virtualenv.rst.txt)  
11. Rerank Models — Xinference, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v1.8.0/models/builtin/rerank/index.html](https://inference.readthedocs.io/en/v1.8.0/models/builtin/rerank/index.html)  
12. Custom Models \- Xinference \- Read the Docs, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/stable/models/custom.html](https://inference.readthedocs.io/en/stable/models/custom.html)  
13. Custom Models \- Xinference, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v1.0.1/models/custom.html](https://inference.readthedocs.io/en/v1.0.1/models/custom.html)  
14. Custom Models \- Xinference \- Read the Docs, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v0.13.3/models/custom.html](https://inference.readthedocs.io/en/v0.13.3/models/custom.html)  
15. Models \- Xinference \- Read the Docs, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v1.1.0/models/index.html](https://inference.readthedocs.io/en/v1.1.0/models/index.html)  
16. Models \- Xinference \- Read the Docs, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v1.5.0.post2/models/index.html](https://inference.readthedocs.io/en/v1.5.0.post2/models/index.html)  
17. Cross-Encoders, ColBERT, and LLM-Based Re-Rankers: A Practical Guide \- Medium, 3月 3, 2026にアクセス、 [https://medium.com/@aimichael/cross-encoders-colbert-and-llm-based-re-rankers-a-practical-guide-a23570d88548](https://medium.com/@aimichael/cross-encoders-colbert-and-llm-based-re-rankers-a-practical-guide-a23570d88548)  
18. Cross-Encoder Reranker: Contextualized Ranking \- Emergent Mind, 3月 3, 2026にアクセス、 [https://www.emergentmind.com/topics/cross-encoder-reranker](https://www.emergentmind.com/topics/cross-encoder-reranker)  
19. Custom Models \- Xinference, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v0.14.4.post1/models/custom.html](https://inference.readthedocs.io/en/v0.14.4.post1/models/custom.html)  
20. Sentence Embeddings. Cross-encoders and Re-ranking – hackerllama \- GitHub Pages, 3月 3, 2026にアクセス、 [https://osanseviero.github.io/hackerllama/blog/posts/sentence\_embeddings2/](https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings2/)  
21. fugashi, a Tool for Tokenizing Japanese in Python \- ACL Anthology, 3月 3, 2026にアクセス、 [https://aclanthology.org/2020.nlposs-1.7.pdf](https://aclanthology.org/2020.nlposs-1.7.pdf)  
22. fugashi \- PyPI, 3月 3, 2026にアクセス、 [https://pypi.org/project/fugashi/](https://pypi.org/project/fugashi/)  
23. Fast Japanese Tokenization with a Single Pip Install \- Dampfkraft, 3月 3, 2026にアクセス、 [https://www.dampfkraft.com/nlp/fast-japanese-tokenizer-for-python.html](https://www.dampfkraft.com/nlp/fast-japanese-tokenizer-for-python.html)  
24. Client API \- Xinference \- Read the Docs, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/stable/user\_guide/client\_api.html](https://inference.readthedocs.io/en/stable/user_guide/client_api.html)  
25. Models \- Xinference \- Read the Docs, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v0.14.1.post1/models/index.html](https://inference.readthedocs.io/en/v0.14.1.post1/models/index.html)  
26. Rerank \- Xinference \- Read the Docs, 3月 3, 2026にアクセス、 [https://inference.readthedocs.io/en/v1.9.0/models/model\_abilities/rerank.html](https://inference.readthedocs.io/en/v1.9.0/models/model_abilities/rerank.html)  
27. Xinference rerank \- LlamaIndex, 3月 3, 2026にアクセス、 [https://developers.llamaindex.ai/python/framework-api-reference/postprocessor/xinference\_rerank/](https://developers.llamaindex.ai/python/framework-api-reference/postprocessor/xinference_rerank/)  
28. Ruri Reranker Small · Models \- Dataloop, 3月 3, 2026にアクセス、 [https://dataloop.ai/library/model/cl-nagoya\_ruri-reranker-small/](https://dataloop.ai/library/model/cl-nagoya_ruri-reranker-small/)  
29. How to add a rerank model bge-reranker-v2-m3? · Issue \#12399 · infiniflow/ragflow \- GitHub, 3月 3, 2026にアクセス、 [https://github.com/infiniflow/ragflow/issues/12399](https://github.com/infiniflow/ragflow/issues/12399)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAaCAYAAABozQZiAAABFUlEQVR4XmNgGAUDD5yA+DoQ/wfiZCAOAeJXUL4QkjoMAJIEKdoIxDJI4nxAvBSI9wAxP5I4HNxigGhkRZdAAiD59+iCINNAEofQJdAASA0IwwEjEDcD8V8gdkWWwAK+MqBpngIViEAWxAH+MaBpfgIVkEYWxAEwnI0hgAOIMEDUvUUWJFazBwNE3XJkQWI0GzNAAms6ELMgS6BrBkWbMxArIIntYoCowUggsKQIijIQ+MQACURYYmCGyoMSEQYApahZQHwFiEuQxEGGgQIHpBFvmgYZ8IsBonARA8Sw81BsjqSOE4mNF/AwQLwkzgBxej4DWmDhAyDNIJd8gWJQ8iUJwGJiNQOWkCYE9BkghQHI2YMEAACYNkNW+e5z5gAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAZCAYAAADXPsWXAAAA4UlEQVR4Xu2TMQoCMRBFx0JQLEQbEWysLC2sPYKNlxHtvYXgKUQrL+ABvIAnELSwUOdnNhK+SbQShH3wYNk/GTKbrEjJT6iqHbUb8WuW6uODa3XgF+S4ii2IMRfLxhwwKLrzy4K6uivEc5SWWJMNBwETsZopB56hWMGMgwDfJFmzUk9qj4MA1CR3glEOYqPUKPP4mos6osyBUc7qgoMALEQDNELDN3AHsM3cKDgV1PQ58KA7ClKjABx96g45cvejre7Vrdqk7EVFrAlOhkF2FMsblDnwcyFMeRP7Dvg5S/6OJ7fuOhnUG7NAAAAAAElFTkSuQmCC>

[image3]: <data:image/png;base64,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>