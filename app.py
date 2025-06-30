import streamlit as st
import json
import pandas as pd
from typing import Dict, List, Any, Optional

@st.cache_data
def load_data() -> Dict[str, Any]:
    """Load and cache the JSON data."""
    with open('final_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model_names(data: Dict[str, Any]) -> List[str]:
    """Extract unique model names from the data and sort them properly."""
    model_names = set()
    for record in data.get('detailed_results', []):
        model_names.add(record['model_name'])
    
    # Custom sorting: base_model first, then checkpoint numbers in ascending order
    model_list = list(model_names)
    
    def sort_key(model_name):
        if model_name == 'base_model':
            return (0, 0)  # base_model comes first
        elif model_name.startswith('checkpoint_'):
            try:
                # Extract number from checkpoint_XXX
                number = int(model_name.split('_')[1])
                return (1, number)  # checkpoints come after base_model, sorted by number
            except (IndexError, ValueError):
                return (2, model_name)  # fallback for unexpected format
        else:
            return (2, model_name)  # other models come last, sorted alphabetically
    
    return sorted(model_list, key=sort_key)

def get_datasets(data: Dict[str, Any]) -> List[str]:
    """Extract unique dataset names from the data."""
    datasets = set()
    for record in data.get('detailed_results', []):
        datasets.add(record['dataset'])
    return sorted(list(datasets))

def get_problem_ids(data: Dict[str, Any]) -> List[int]:
    """Extract unique problem IDs from the data."""
    problem_ids = set()
    for record in data.get('detailed_results', []):
        problem_ids.add(record['problem_id'])
    return sorted(list(problem_ids))

def filter_records(data: Dict[str, Any], model1: str, model2: str, dataset: str, problem_id: int) -> tuple:
    """Filter records based on selection criteria."""
    records = data.get('detailed_results', [])
    
    record1 = None
    record2 = None
    
    for record in records:
        if (record['model_name'] == model1 and 
            record['dataset'] == dataset and 
            record['problem_id'] == problem_id):
            record1 = record
        elif (record['model_name'] == model2 and 
              record['dataset'] == dataset and 
              record['problem_id'] == problem_id):
            record2 = record
    
    return record1, record2

def calculate_is_correct(predicted_answer: Any, true_answer: Any) -> bool:
    """Calculate is_correct if not present in data."""
    try:
        return float(predicted_answer) == float(true_answer)
    except (ValueError, TypeError):
        return str(predicted_answer).strip() == str(true_answer).strip()

def render_normal_view(record1: Dict[str, Any], record2: Dict[str, Any], model1: str, model2: str):
    """Render the normal view for non-elyza datasets."""
    st.subheader("比較結果")
    
    # Question
    st.write("**Question:**")
    st.write(record1.get('question', 'N/A'))
    
    # True Answer - different height for custom dataset
    st.write("**True Answer:**")
    dataset = record1.get('dataset', '')
    true_answer_height = 450 if dataset == 'custom' else 70
    st.text_area("", value=str(record1.get('true_answer', 'N/A')), height=true_answer_height, key="true_answer", disabled=True)
    
    # Create two columns for model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"## {model1}")
        
        # Predicted Answer
        st.write("**Predicted Answer:**")
        st.text_area("", value=str(record1.get('predicted_answer', 'N/A')), height=70, key=f"predicted1_{model1}", disabled=True)
        
        # Is Correct
        is_correct1 = record1.get('is_correct')
        if is_correct1 is None:
            is_correct1 = calculate_is_correct(record1['predicted_answer'], record1['true_answer'])
        
        if is_correct1:
            st.success("✅ Correct")
        else:
            st.error("❌ Incorrect")
        
        # Model Response
        st.write("**Model Response:**")
        st.text_area("", value=record1.get('model_response', 'N/A'), height=600, key=f"response1_{model1}")
    
    with col2:
        st.markdown(f"## {model2}")
        
        # Predicted Answer
        st.write("**Predicted Answer:**")
        st.text_area("", value=str(record2.get('predicted_answer', 'N/A')), height=70, key=f"predicted2_{model2}", disabled=True)
        
        # Is Correct
        is_correct2 = record2.get('is_correct')
        if is_correct2 is None:
            is_correct2 = calculate_is_correct(record2['predicted_answer'], record2['true_answer'])
        
        if is_correct2:
            st.success("✅ Correct")
        else:
            st.error("❌ Incorrect")
        
        # Model Response
        st.write("**Model Response:**")
        st.text_area("", value=record2.get('model_response', 'N/A'), height=600, key=f"response2_{model2}")

def render_elyza_view(record1: Dict[str, Any], record2: Dict[str, Any], model1: str, model2: str):
    """Render the elyza view for elyza dataset."""
    st.subheader("比較結果 (ELYZA)")
    
    # Question
    st.write("**Question:**")
    st.write(record1.get('question', 'N/A'))
    
    # Eval Aspect
    st.write("**Evaluation Aspect:**")
    st.write(record1.get('eval_aspect', 'N/A'))
    
    # Create two columns for model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"## {model1}")
        
        # GPT4 Score
        score1 = record1.get('gpt4_score', 0)
        score_color1 = get_score_color(score1)
        st.markdown(f"**GPT4 Score:** <span style='color: {score_color1}; font-weight: bold;'>{score1}/5</span>", unsafe_allow_html=True)
        
        # Model Response
        st.write("**Model Response:**")
        st.text_area("", value=record1.get('model_response', 'N/A'), height=600, key=f"elyza_response1_{model1}")
    
    with col2:
        st.markdown(f"## {model2}")
        
        # GPT4 Score
        score2 = record2.get('gpt4_score', 0)
        score_color2 = get_score_color(score2)
        st.markdown(f"**GPT4 Score:** <span style='color: {score_color2}; font-weight: bold;'>{score2}/5</span>", unsafe_allow_html=True)
        
        # Model Response
        st.write("**Model Response:**")
        st.text_area("", value=record2.get('model_response', 'N/A'), height=600, key=f"elyza_response2_{model2}")

def get_score_color(score: float) -> str:
    """Get color based on score (0-5 scale)."""
    if score >= 4.5:
        return "#00FF00"  # Green
    elif score >= 3.5:
        return "#7FFF00"  # Light green
    elif score >= 2.5:
        return "#FFFF00"  # Yellow
    elif score >= 1.5:
        return "#FF7F00"  # Orange
    else:
        return "#FF0000"  # Red

def main():
    st.set_page_config(
        page_title="🚀 AI Model Battle Arena",
        page_icon="🚀",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Selection panel styling */
    .selection-panel {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    
    /* Comparison cards */
    .comparison-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Model headers */
    .model-header {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    
    /* Question section */
    .question-section {
        background-color: #f1f3f4;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #4285f4;
        margin-bottom: 2rem;
    }
    
    /* Answer sections */
    .answer-section {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 6px;
        border-left: 3px solid #ffc107;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-correct {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid #c3e6cb;
        display: inline-block;
        font-weight: bold;
    }
    
    .status-incorrect {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid #f5c6cb;
        display: inline-block;
        font-weight: bold;
    }
    
    /* Score styling */
    .score-display {
        background: linear-gradient(135deg, #a8e6cf 0%, #88d8a3 100%);
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        color: #2d3436;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #4285f4;
        box-shadow: 0 0 0 2px rgba(66, 133, 244, 0.2);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
    }
    
    /* Column spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Sticky selection panel - using direct CSS injection */
    .stApp > div:first-child > div.sticky-controls {
        position: sticky !important;
        top: 0 !important;
        background-color: white !important;
        z-index: 999 !important;
        padding: 1rem 0 !important;
        border-bottom: 2px solid #e9ecef !important;
        margin-bottom: 1rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Alternative approach using element targeting */
    div[data-testid="stSidebar"] + div > div:first-child > div:nth-child(3) {
        position: sticky !important;
        top: 0 !important;
        background-color: white !important;
        z-index: 999 !important;
        padding: 1rem 0 !important;
        border-bottom: 2px solid #e9ecef !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header with enhanced styling
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">🚀 AI Model Battle Arena</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.2rem; opacity: 0.9;">AIモデル対戦アリーナ - 次世代モデル評価プラットフォーム</p>
        <p style="margin: 0.3rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">⚡ 最先端AIモデルの性能を直接対決で比較する革新的システム ⚡</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    try:
        data = load_data()
    except FileNotFoundError:
        st.error("final_results.json ファイルが見つかりません。")
        st.stop()
    except json.JSONDecodeError:
        st.error("JSON ファイルの読み込みに失敗しました。")
        st.stop()
    
    # Get available options
    model_names = get_model_names(data)
    datasets = get_datasets(data)
    problem_ids = get_problem_ids(data)
    
    # Selection interface with container for sticky positioning
    with st.container():
        st.markdown('<div class="sticky-controls">', unsafe_allow_html=True)
        st.subheader("⚔️ バトル設定")
        
        col1, col2, col3, col4 = st.columns(4)
    
        with col1:
            model1 = st.selectbox(
                "Model 1",
                options=model_names,
                key="model1",
                help="比較対象のモデル1を選択"
            )
        
        with col2:
            model2 = st.selectbox(
                "Model 2", 
                options=model_names,
                key="model2",
                help="比較対象のモデル2を選択"
            )
        
        with col3:
            dataset = st.selectbox(
                "Dataset",
                options=datasets,
                key="dataset",
                help="評価データセットを選択"
            )
        
        with col4:
            problem_id = st.selectbox(
                "Problem ID",
                options=problem_ids,
                key="problem_id",
                help="問題IDを選択 (0-49)"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Filter and display results
    if model1 and model2 and dataset is not None and problem_id is not None:
        # Check if same model is selected
        if model1 == model2:
            st.error("同じモデルが選択されています。違うモデルを選んでください。")
        else:
            record1, record2 = filter_records(data, model1, model2, dataset, problem_id)
            
            if record1 is None or record2 is None:
                st.error("該当するデータが見つかりません。条件を確認してください。")
                
                if record1 is None:
                    st.warning(f"Model: {model1}, Dataset: {dataset}, Problem ID: {problem_id} のデータが見つかりません。")
                if record2 is None:
                    st.warning(f"Model: {model2}, Dataset: {dataset}, Problem ID: {problem_id} のデータが見つかりません。")
            else:
                # Render appropriate view based on dataset
                if dataset == "elyza":
                    render_elyza_view(record1, record2, model1, model2)
                else:
                    render_normal_view(record1, record2, model1, model2)

if __name__ == "__main__":
    main()