import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import anthropic
import openai
import cohere
from groq import Groq

from src.core.proctor import HLEProctor
from src.models.evaluation import Question, Response
from src.utils.visualization import plot_performance_metrics, plot_confidence_calibration

# Initialize clients
@st.cache_resource
def get_llm_clients():
    """Initialize all available LLM clients."""
    clients = {}
    
    # Anthropic (Claude)
    if os.getenv('ANTHROPIC_API_KEY'):
        clients['claude-3-opus'] = {'client': anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')), 'model': 'claude-3-opus-20240229'}
        clients['claude-3-sonnet'] = {'client': anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')), 'model': 'claude-3-sonnet-20240229'}
        clients['claude-2.1'] = {'client': anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')), 'model': 'claude-2.1'}
    
    # OpenAI
    if os.getenv('OPENAI_API_KEY'):
        openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        clients['gpt-4-turbo'] = {'client': openai_client, 'model': 'gpt-4-0125-preview'}
        clients['gpt-4'] = {'client': openai_client, 'model': 'gpt-4'}
        clients['gpt-3.5-turbo'] = {'client': openai_client, 'model': 'gpt-3.5-turbo'}
    
    # Cohere
    if os.getenv('COHERE_API_KEY'):
        cohere_client = cohere.Client(os.getenv('COHERE_API_KEY'))
        clients['cohere-command'] = {'client': cohere_client, 'model': 'command'}
        clients['cohere-command-light'] = {'client': cohere_client, 'model': 'command-light'}
    
    # Groq
    if os.getenv('GROQ_API_KEY'):
        groq_client = Groq(api_key=os.getenv('GROQ_API_KEY'))
        clients['mixtral-8x7b'] = {'client': groq_client, 'model': 'mixtral-8x7b-v0.1'}
        clients['llama2-70b'] = {'client': groq_client, 'model': 'llama2-70b-4096'}
    
    return clients

def get_llm_answer(client_name: str, client_info: dict, question: Question) -> tuple[str, float, str]:
    """Get answer, confidence, and reasoning from an LLM."""
    try:
        system_prompt = """You are an expert AI assistant taking a challenging exam. For each question:
1. Carefully analyze the problem
2. Show your step-by-step reasoning
3. Provide your final answer
4. Rate your confidence from 0.0 to 1.0

Format your response exactly as follows:
Reasoning: <your detailed step-by-step reasoning>
Answer: <your final answer>
Confidence: <confidence score between 0.0 and 1.0>"""

        question_prompt = f"Question: {question.content}\n\nProvide your response in the exact format specified."
        
        if 'claude' in client_name:
            response = client_info['client'].messages.create(
                model=client_info['model'],
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": question_prompt}]
            )
            response_text = response.content[0].text
            
        elif 'gpt' in client_name:
            response = client_info['client'].chat.completions.create(
                model=client_info['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question_prompt}
                ]
            )
            response_text = response.choices[0].message.content
            
        elif 'cohere' in client_name:
            response = client_info['client'].generate(
                model=client_info['model'],
                prompt=f"{system_prompt}\n\n{question_prompt}",
                max_tokens=1000
            )
            response_text = response.generations[0].text
            
        elif 'mixtral' in client_name or 'llama2' in client_name:
            response = client_info['client'].chat.completions.create(
                model=client_info['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question_prompt}
                ]
            )
            response_text = response.choices[0].message.content
            
        # Extract answer, confidence, and reasoning
        parts = response_text.split('\n')
        reasoning = ""
        answer = ""
        confidence = 0.0
        
        for part in parts:
            if part.startswith('Reasoning:'):
                reasoning = part[10:].strip()
            elif part.startswith('Answer:'):
                answer = part[7:].strip()
            elif part.startswith('Confidence:'):
                try:
                    confidence = float(part[11:].strip())
                except:
                    confidence = 0.0
        
        return answer, confidence, reasoning
        
    except Exception as e:
        return "", 0.0, f"Error: {str(e)}"

# Initialize the proctor
@st.cache_resource
def get_proctor():
    return HLEProctor()

# Page configuration
st.set_page_config(
    page_title="HLE Proctor AI",
    page_icon="🎓",
    layout="wide"
)

# Title and description
st.title("🎓 HLE Proctor AI")
st.markdown("""
This application evaluates Large Language Models (LLMs) using the Humanity's Last Exam (HLE) benchmark.
It systematically administers questions, monitors responses, and generates insights about model performance.
""")

# Initialize LLM clients
llm_clients = get_llm_clients()
if not llm_clients:
    st.warning("""
    ⚠️ No LLM API keys found. To enable AI functionality, add your API keys to the .env file:
    ```
    ANTHROPIC_API_KEY="your-key-here"
    OPENAI_API_KEY="your-key-here"
    COHERE_API_KEY="your-key-here"
    GROQ_API_KEY="your-key-here"
    ```
    """)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'questions_answered' not in st.session_state:
    st.session_state.questions_answered = 0
if 'results' not in st.session_state:
    st.session_state.results = []
if 'llm_results' not in st.session_state:
    st.session_state.llm_results = {}

# Sidebar for session control
with st.sidebar:
    st.header("Session Control")
    llm_name = st.text_input("Your LLM Name", placeholder="e.g., GPT-4, Claude, etc.")
    max_questions = st.number_input("Maximum Questions", min_value=1, max_value=3000, value=100)
    
    # LLM Selection with categories
    st.subheader("🤖 Competing LLMs")
    selected_llms = {}
    
    # Anthropic Models
    if any('claude' in k for k in llm_clients.keys()):
        st.markdown("##### Anthropic")
        for llm in [k for k in llm_clients.keys() if 'claude' in k]:
            selected_llms[llm] = st.checkbox(f"Enable {llm.replace('-', ' ').title()}", value=True)
    
    # OpenAI Models
    if any('gpt' in k for k in llm_clients.keys()):
        st.markdown("##### OpenAI")
        for llm in [k for k in llm_clients.keys() if 'gpt' in k]:
            selected_llms[llm] = st.checkbox(f"Enable {llm.replace('-', ' ').title()}", value=True)
    
    # Cohere Models
    if any('cohere' in k for k in llm_clients.keys()):
        st.markdown("##### Cohere")
        for llm in [k for k in llm_clients.keys() if 'cohere' in k]:
            selected_llms[llm] = st.checkbox(f"Enable {llm.replace('-', ' ').title()}", value=True)
    
    # Groq Models
    if any(k in ['mixtral-8x7b', 'llama2-70b'] for k in llm_clients.keys()):
        st.markdown("##### Groq")
        for llm in [k for k in llm_clients.keys() if k in ['mixtral-8x7b', 'llama2-70b']]:
            selected_llms[llm] = st.checkbox(f"Enable {llm.replace('-', ' ').title()}", value=True)
    
    if st.button("Start New Session"):
        proctor = get_proctor()
        st.session_state.session_id = proctor.start_session(llm_name, max_questions)
        st.session_state.questions_answered = 0
        st.session_state.results = []
        st.session_state.llm_results = {}
        st.success(f"Session started! ID: {st.session_state.session_id}")

# Main content area
if st.session_state.session_id:
    # Create columns for layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Question Area")
        proctor = get_proctor()
        
        # Get next question if needed
        if not st.session_state.current_question:
            st.session_state.current_question = proctor.get_next_question()
        
        if st.session_state.current_question:
            question = st.session_state.current_question
            st.subheader(f"Question {st.session_state.questions_answered + 1}")
            st.markdown(f"**Subject Area:** {question.subject_area}")
            st.markdown(f"**Difficulty:** {question.difficulty}")
            st.markdown("---")
            st.markdown(question.content)
            
            # Get LLM answers first
            if question.id not in st.session_state.llm_results:
                st.session_state.llm_results[question.id] = {}
                
                with st.spinner("Getting LLM answers..."):
                    for llm_name, client_info in llm_clients.items():
                        if selected_llms.get(llm_name, False):
                            answer, confidence, reasoning = get_llm_answer(llm_name, client_info, question)
                            st.session_state.llm_results[question.id][llm_name] = {
                                'answer': answer,
                                'confidence': confidence,
                                'reasoning': reasoning
                            }
            
            # Show LLM answers in an expander
            with st.expander("View LLM Answers", expanded=True):
                for llm_name, result in st.session_state.llm_results[question.id].items():
                    st.markdown(f"### {llm_name.replace('-', ' ').title()}")
                    st.markdown(f"**Answer:** {result['answer']}")
                    st.markdown(f"**Confidence:** {result['confidence']*100:.0f}%")
                    st.markdown(f"**Reasoning:**\n{result['reasoning']}")
                    st.markdown("---")
            
            # Handle user answer
            if question.type == "multiple_choice" and question.options:
                answer = st.radio("Select your answer:", question.options)
            else:
                answer = st.text_area("Your answer:", height=100, key="answer_input")
            
            confidence = st.slider("Confidence Level (0-100%)", 0, 100, 50) / 100
            
            # Add validation
            submit_disabled = not answer
            
            if submit_disabled:
                st.warning("Please provide an answer before submitting.")
            
            if st.button("Submit Answer", disabled=submit_disabled):
                # Use the best performing LLM's reasoning
                best_llm = None
                best_confidence = -1
                
                for llm_name, result in st.session_state.llm_results[question.id].items():
                    if result['confidence'] > best_confidence:
                        best_confidence = result['confidence']
                        best_llm = llm_name
                
                reasoning = st.session_state.llm_results[question.id][best_llm]['reasoning'] if best_llm else "No LLM reasoning available."
                
                st.markdown("### Selected Reasoning")
                st.markdown(reasoning)
                
                is_correct, details = proctor.submit_response(
                    question.id,
                    answer,
                    confidence,
                    reasoning
                )
                
                # Show feedback
                if is_correct:
                    st.success("✅ Correct!")
                else:
                    st.error("❌ Incorrect")
                    st.markdown(f"**Correct Answer:** {details['correct_answer']}")
                
                if details['explanation']:
                    st.markdown("**Explanation:**")
                    st.markdown(details['explanation'])
                
                # Store result
                st.session_state.results.append({
                    'question_id': question.id,
                    'is_correct': is_correct,
                    'confidence': confidence,
                    'details': details
                })
                
                st.session_state.questions_answered += 1
                st.session_state.current_question = None
                
                # Add a continue button
                if st.button("Next Question"):
                    st.rerun()
        else:
            st.info("No more questions available.")
    
    with col2:
        st.header("Session Statistics")
        if st.session_state.results:
            # Calculate basic statistics
            correct_answers = sum(1 for r in st.session_state.results if r['is_correct'])
            accuracy = correct_answers / len(st.session_state.results)
            avg_confidence = sum(r['confidence'] for r in st.session_state.results) / len(st.session_state.results)
            
            # Display metrics
            st.metric("Questions Answered", st.session_state.questions_answered)
            st.metric("Your Accuracy", f"{accuracy:.2%}")
            st.metric("Your Avg Confidence", f"{avg_confidence:.2%}")
            
            # Calculate LLM statistics
            if st.session_state.llm_results:
                st.subheader("LLM Performance")
                for llm_name in llm_clients.keys():
                    if not selected_llms.get(llm_name, False):
                        continue
                        
                    llm_correct = 0
                    llm_total = 0
                    llm_confidence = 0
                    
                    for qid, results in st.session_state.llm_results.items():
                        if llm_name in results:
                            llm_total += 1
                            llm_confidence += results[llm_name]['confidence']
                            # We need to check if the LLM's answer matches the correct answer
                            correct_answer = next((r['details']['correct_answer'] for r in st.session_state.results if r['question_id'] == qid), None)
                            if correct_answer and results[llm_name]['answer'].strip().lower() == correct_answer.strip().lower():
                                llm_correct += 1
                    
                    if llm_total > 0:
                        llm_accuracy = llm_correct / llm_total
                        llm_avg_confidence = llm_confidence / llm_total
                        st.metric(f"{llm_name.replace('-', ' ').title()} Accuracy", f"{llm_accuracy:.2%}")
                        st.metric(f"{llm_name.replace('-', ' ').title()} Avg Confidence", f"{llm_avg_confidence:.2%}")
            
            # Create a progress chart
            progress_df = pd.DataFrame(st.session_state.results)
            st.line_chart(progress_df['is_correct'].rolling(window=5).mean())
            
            if st.button("Generate Full Report"):
                report = proctor.generate_report()
                
                # Display report visualizations
                st.subheader("Performance Analysis")
                fig = plot_performance_metrics(report.metrics.dict())
                st.pyplot(fig)
                
                # Export results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"results/evaluation_{timestamp}.json"
                proctor.export_results(export_path)
                st.success(f"Results exported to {export_path}")
else:
    st.info("👈 Start a new session using the sidebar controls")

# Footer
st.markdown("---")
st.markdown("*HLE Proctor AI - Evaluating the Next Generation of Language Models*")
