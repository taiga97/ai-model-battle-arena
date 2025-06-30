# 🚀 AI Model Battle Arena
**AIモデル対戦アリーナ - 次世代モデル評価プラットフォーム**

最先端のAIモデル性能を直接対決で比較する革新的なビジュアライゼーションシステム

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/taiga97/ai-model-battle-arena)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io/)


## 機能

- **モデル比較**: 2つのモデルの出力を並列比較
- **データセット対応**: gsm8k, openr1_math, custom, elyza データセットをサポート
- **動的ビュー切替**: elyzaデータセットでは専用ビューを表示
- **レスポンシブデザイン**: 画面サイズに応じてレイアウトが調整

## セットアップ

1. 依存関係をインストール:
```bash
pip install -r requirements.txt
```

2. アプリケーションを起動:
```bash
streamlit run app.py
```

## 使用方法

1. **Model 1** と **Model 2** を選択
2. **Dataset** を選択 (gsm8k, openr1_math, custom, elyza)
3. **Problem ID** を選択 (0-49)
4. 選択条件に基づいて結果が自動表示

## データ形式

`final_results.json` が必要です。以下の構造を想定:

```json
{
  "detailed_results": [
    {
      "model_name": "base_model",
      "dataset": "gsm8k", 
      "problem_id": 0,
      "question": "問題文",
      "true_answer": "正解",
      "predicted_answer": "予測回答",
      "model_response": "推論過程",
      "is_correct": true
    }
  ]
}
```

elyzaデータセットの場合は追加で:
- `eval_aspect`: 評価観点
- `gpt4_score`: GPT4スコア (0-5)