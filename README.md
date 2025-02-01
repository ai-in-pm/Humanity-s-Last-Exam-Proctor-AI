# HLE Proctor AI

HLE Proctor AI is an advanced AI agent designed to evaluate Large Language Models (LLMs) using the Humanity's Last Exam (HLE) benchmark. This system provides comprehensive evaluation, monitoring, and analysis of LLM performance across various domains.

The development of this repository was inspired by the "Humanity's Last Exam" paper. To read the full paper, visit https://arxiv.org/pdf/2501.14249

## Features

- **Exam Administration**: Systematically presents 3,000 questions from the HLE dataset
- **Multi-Modal Support**: Handles both text and image-based questions
- **Automated Evaluation**: Scores responses and analyzes reasoning processes
- **Confidence Calibration**: Tracks model confidence and detects hallucinations
- **Performance Analytics**: Generates detailed reports and insights
- **Anti-Gaming Measures**: Prevents external resource usage and detects memorization

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/hle-proctor-ai.git
cd hle-proctor-ai
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
COHERE_API_KEY=your_key_here
EMERGENCEAI_API_KEY=your_key_here
```

## Usage

1. Start the evaluation server:

```bash
python -m src.main
```

2. Access the dashboard at `http://localhost:8000`
3. Configure your evaluation settings and select the LLM to evaluate
4. Start the evaluation process

## Project Structure

```
hle-proctor-ai/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── evaluation.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── proctor.py
│   │   ├── evaluator.py
│   │   └── question_bank.py
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py
│       └── visualization.py
├── tests/
│   ├── __init__.py
│   └── test_proctor.py
├── data/
│   └── questions/
├── .env
├── requirements.txt
└── README.md
```

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- HLE Benchmark creators and contributors
- The AI research community for valuable insights and feedback
