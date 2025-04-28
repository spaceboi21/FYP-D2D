# LLM Contextual Understanding Comparison

This project compares the contextual understanding capabilities of three leading large language models:
- OpenAI's GPT-4
- Anthropic's Claude 3 Opus
- Meta's Llama 3 70B

## Overview

The comparison focuses on six key scenarios that test different aspects of contextual understanding:

1. **Ambiguous Pronoun Resolution** - Ability to resolve ambiguous pronouns in sentences
2. **Causal Reasoning** - Understanding cause and effect relationships
3. **Temporal Understanding** - Comprehending sequence of events
4. **Counterfactual Reasoning** - Processing hypothetical situations
5. **Multi-hop Inference** - Making multi-step logical deductions
6. **Contextual Nuance** - Understanding subtle context in complex scenarios

## Evaluation Criteria

Models are evaluated on four criteria:
- **Accuracy** (40%) - Correctness of the response
- **Reasoning** (30%) - Quality of the reasoning process
- **Context Use** (20%) - How well the model uses provided context
- **Nuance** (10%) - Handling of subtlety and edge cases

## Visualizations

The script generates five visualization files:

1. **model_comparison_overall.png** - Bar chart showing overall weighted scores by model and scenario
2. **model_radar_comparison.html** - Interactive radar chart comparing models across evaluation criteria
3. **model_scenario_heatmap.png** - Heatmap displaying model performance across different scenarios
4. **model_complexity_relationship.png** - Scatter plot with trendlines showing performance vs. task complexity
5. **criteria_breakdown.html** - Interactive detailed breakdown of each evaluation criterion

## Key Findings

Based on the hardcoded evaluation data:

- **GPT-4** achieved the highest overall score (8.78/10), showing particular strength in temporal understanding and multi-hop inference.
- **Claude 3 Opus** performed very similarly (8.73/10), with specific advantages in counterfactual reasoning and contextual nuance.
- **Llama 3 70B** scored lower overall (7.38/10) but still demonstrated strong capabilities in temporal understanding and multi-hop inference.

All models tend to perform better on simpler tasks (complexity levels 2-3) and show reduced performance on more complex tasks (levels 4-5).

## How to Run

```bash
python llm_comparison.py
```

This will generate all visualization files and display a summary of results in the terminal.

## Dependencies

- Python 3.x
- pandas
- matplotlib
- seaborn
- plotly
- numpy
- python-dotenv

## Note

The current implementation uses hardcoded evaluation data. For a live API-based comparison, you would need to uncomment and configure the API client sections and implement functions to query each LLM. 