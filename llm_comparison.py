import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from typing import List, Dict, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load environment variables
load_dotenv()

# Test scenarios with different contextual understanding challenges
test_scenarios = [
    {
        "name": "Ambiguous Pronoun Resolution",
        "prompt": "John told Bob that he had won the competition. Who won the competition?",
        "context": "This tests the model's ability to handle ambiguous pronouns.",
        "complexity": 3
    },
    {
        "name": "Causal Reasoning",
        "prompt": "The trophy doesn't fit in the suitcase because it's too large. What is too large?",
        "context": "This tests if the model can correctly identify what 'it' refers to.",
        "complexity": 4
    },
    {
        "name": "Temporal Understanding",
        "prompt": "Before going to the store, Sarah finished her homework. What did Sarah do first?",
        "context": "This tests understanding of sequence of events.",
        "complexity": 2
    },
    {
        "name": "Counterfactual Reasoning",
        "prompt": "If the electricity hadn't gone out, we would have watched a movie. Did we watch a movie?",
        "context": "This tests understanding of hypothetical situations.",
        "complexity": 5
    },
    {
        "name": "Multi-hop Inference",
        "prompt": "All mammals have fur. Some animals without fur are reptiles. Is a snake a mammal?",
        "context": "This tests multi-step logical reasoning.",
        "complexity": 4
    },
    {
        "name": "Contextual Nuance",
        "prompt": "The city council refused the demonstrators a permit because they feared violence. Who feared violence?",
        "context": "This tests understanding of complex social scenarios.",
        "complexity": 5
    },
]

# Evaluation criteria and weights
evaluation_criteria = {
    "accuracy": 0.4,       # Correctness of the response
    "reasoning": 0.3,      # Quality of reasoning process
    "context_use": 0.2,    # How well the model uses provided context
    "nuance": 0.1          # Handling of subtlety and edge cases
}

# Hardcoded evaluation results to simulate actual API calls
# Scores are on a scale of 1-10 for each criterion
hardcoded_results = {
    "GPT-4": {
        "Ambiguous Pronoun Resolution": {"accuracy": 9, "reasoning": 8, "context_use": 9, "nuance": 8},
        "Causal Reasoning": {"accuracy": 9, "reasoning": 9, "context_use": 8, "nuance": 9},
        "Temporal Understanding": {"accuracy": 10, "reasoning": 9, "context_use": 9, "nuance": 8},
        "Counterfactual Reasoning": {"accuracy": 8, "reasoning": 9, "context_use": 8, "nuance": 9},
        "Multi-hop Inference": {"accuracy": 9, "reasoning": 10, "context_use": 9, "nuance": 8},
        "Contextual Nuance": {"accuracy": 8, "reasoning": 9, "context_use": 8, "nuance": 9},
    },
    "Claude 3 Opus": {
        "Ambiguous Pronoun Resolution": {"accuracy": 9, "reasoning": 9, "context_use": 8, "nuance": 9},
        "Causal Reasoning": {"accuracy": 8, "reasoning": 9, "context_use": 8, "nuance": 9},
        "Temporal Understanding": {"accuracy": 9, "reasoning": 8, "context_use": 9, "nuance": 8},
        "Counterfactual Reasoning": {"accuracy": 9, "reasoning": 10, "context_use": 9, "nuance": 8},
        "Multi-hop Inference": {"accuracy": 8, "reasoning": 9, "context_use": 9, "nuance": 8},
        "Contextual Nuance": {"accuracy": 9, "reasoning": 9, "context_use": 8, "nuance": 10},
    },
    "Llama 3 70B": {
        "Ambiguous Pronoun Resolution": {"accuracy": 8, "reasoning": 7, "context_use": 7, "nuance": 7},
        "Causal Reasoning": {"accuracy": 7, "reasoning": 8, "context_use": 7, "nuance": 7},
        "Temporal Understanding": {"accuracy": 9, "reasoning": 8, "context_use": 7, "nuance": 7},
        "Counterfactual Reasoning": {"accuracy": 7, "reasoning": 7, "context_use": 6, "nuance": 6},
        "Multi-hop Inference": {"accuracy": 8, "reasoning": 9, "context_use": 7, "nuance": 7},
        "Contextual Nuance": {"accuracy": 7, "reasoning": 7, "context_use": 6, "nuance": 7},
    }
}

# Function to calculate weighted scores
def calculate_weighted_score(criteria_scores: Dict[str, int]) -> float:
    """Calculate weighted score based on evaluation criteria weights"""
    return sum(criteria_scores[criterion] * weight 
               for criterion, weight in evaluation_criteria.items())

# Process the results to get a DataFrame for visualization
def process_results() -> pd.DataFrame:
    """Process hardcoded results into a pandas DataFrame"""
    data = []
    
    for model_name, scenarios in hardcoded_results.items():
        for scenario_name, criteria_scores in scenarios.items():
            # Get the complexity level from test_scenarios
            scenario = next((s for s in test_scenarios if s["name"] == scenario_name), None)
            complexity = scenario["complexity"] if scenario else 3
            
            # Calculate weighted score
            weighted_score = calculate_weighted_score(criteria_scores)
            
            # Add row for each criteria
            for criterion, score in criteria_scores.items():
                data.append({
                    "Model": model_name,
                    "Scenario": scenario_name,
                    "Criterion": criterion,
                    "Score": score,
                    "Weighted Score": weighted_score,
                    "Complexity": complexity
                })
    
    return pd.DataFrame(data)

# Visualizations
def create_visualizations(df: pd.DataFrame):
    """Create various visualizations comparing model performance"""
    
    # Set the style
    plt.style.use('fivethirtyeight')
    sns.set_palette("deep")
    
    # 1. Overall weighted scores by model
    fig1 = plt.figure(figsize=(12, 6))
    model_scores = df.groupby(['Model', 'Scenario'])['Weighted Score'].first().reset_index()
    sns.barplot(data=model_scores, x='Model', y='Weighted Score', hue='Scenario')
    plt.title('Overall Weighted Scores by Model and Scenario', fontsize=15)
    plt.xticks(rotation=0)
    plt.ylabel('Weighted Score (0-10)')
    plt.legend(title='Scenario', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('model_comparison_overall.png', dpi=300)
    
    # 2. Radar chart comparison using Plotly
    # Calculate average scores for each model and criterion
    radar_df = df.groupby(['Model', 'Criterion'])['Score'].mean().reset_index()
    
    # Create radar chart
    fig = go.Figure()
    
    for model in radar_df['Model'].unique():
        model_data = radar_df[radar_df['Model'] == model]
        fig.add_trace(go.Scatterpolar(
            r=model_data['Score'].values,
            theta=model_data['Criterion'].values,
            fill='toself',
            name=model
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )
        ),
        title="Model Comparison by Evaluation Criteria",
        showlegend=True
    )
    
    fig.write_html('model_radar_comparison.html')
    
    # 3. Heatmap of model performance across scenarios
    pivot_df = df.pivot_table(
        index='Model', 
        columns='Scenario', 
        values='Weighted Score',
        aggfunc='first'
    )
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.1f', linewidths=.5)
    plt.title('Model Performance Across Different Scenarios', fontsize=15)
    plt.tight_layout()
    plt.savefig('model_scenario_heatmap.png', dpi=300)
    
    # 4. Performance vs. Complexity scatter plot with trendlines
    complexity_df = df.groupby(['Model', 'Scenario', 'Complexity'])['Weighted Score'].first().reset_index()
    
    plt.figure(figsize=(12, 8))
    for model in complexity_df['Model'].unique():
        model_data = complexity_df[complexity_df['Model'] == model]
        sns.regplot(
            x='Complexity', 
            y='Weighted Score', 
            data=model_data, 
            label=model,
            scatter_kws={'s': 80}
        )
    
    plt.title('Model Performance vs. Task Complexity', fontsize=15)
    plt.xlabel('Task Complexity (1-5 scale)')
    plt.ylabel('Weighted Score (0-10)')
    plt.legend(title='Model')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_complexity_relationship.png', dpi=300)
    
    # 5. Detailed criteria breakdown by model (interactive)
    criteria_df = df.pivot_table(
        index=['Model', 'Scenario'],
        columns='Criterion',
        values='Score'
    ).reset_index()
    
    # Create subplots for each criterion
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=list(evaluation_criteria.keys()),
                        shared_yaxes=True)
    
    colors = px.colors.qualitative.Plotly
    
    # Add traces for each criterion
    for i, criterion in enumerate(evaluation_criteria.keys()):
        row, col = i // 2 + 1, i % 2 + 1
        
        for j, model in enumerate(criteria_df['Model'].unique()):
            model_data = criteria_df[criteria_df['Model'] == model]
            
            fig.add_trace(
                go.Bar(
                    x=model_data['Scenario'],
                    y=model_data[criterion],
                    name=model,
                    marker_color=colors[j % len(colors)],
                    showlegend=True if i == 0 else False
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="Detailed Criteria Breakdown by Model and Scenario",
        height=800,
        width=1200,
        barmode='group'
    )
    
    fig.write_html('criteria_breakdown.html')
    
    print("Visualizations created successfully!")
    
    # Return HTML files for display
    return 'model_radar_comparison.html', 'criteria_breakdown.html'

# Run the analysis
def main():
    print("LLM Contextual Understanding Comparison")
    print("=======================================")
    
    # Process results
    results_df = process_results()
    
    # Display summary statistics
    model_summary = results_df.groupby('Model')['Weighted Score'].mean().reset_index()
    model_summary = model_summary.sort_values('Weighted Score', ascending=False)
    
    print("\nModel Performance Summary:")
    print("-------------------------")
    for i, row in model_summary.iterrows():
        print(f"{row['Model']}: {row['Weighted Score']:.2f}/10")
    
    # Create visualizations
    html_files = create_visualizations(results_df)
    
    print("\nVisualization files created:")
    print("1. model_comparison_overall.png - Bar chart of overall scores")
    print("2. model_radar_comparison.html - Interactive radar chart")
    print("3. model_scenario_heatmap.png - Heatmap of performance across scenarios")
    print("4. model_complexity_relationship.png - Scatter plot of performance vs. complexity")
    print("5. criteria_breakdown.html - Interactive detailed criteria breakdown")
    
    return html_files

if __name__ == "__main__":
    main() 