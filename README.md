# PAS2 - Hallucination Detection System

A sophisticated system for detecting hallucinations in AI responses using a paraphrase-based approach with model-as-judge verification.

## Features

- **Paraphrase Generation**: Automatically generates semantically equivalent variations of user queries
- **Multi-Model Architecture**: Uses Mistral Large for responses and OpenAI's o3-mini as a judge
- **Real-time Progress Tracking**: Visual feedback during the analysis process
- **Persistent Feedback Storage**: User feedback and results are stored in a persistent SQLite database
- **Interactive Web Interface**: Clean, responsive Gradio interface with example queries
- **Detailed Analysis**: Provides confidence scores, reasoning, and specific conflicting facts
- **Statistics Dashboard**: Real-time tracking of hallucination detection statistics

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your API keys as environment variables:
   - `HF_MISTRAL_API_KEY`: Your Mistral AI API key
   - `HF_OPENAI_API_KEY`: Your OpenAI API key

## Deployment on Hugging Face Spaces

1. Create a new Space on Hugging Face
2. Select "Gradio" as the SDK
3. Add your repository
4. Set the following secrets in your Space's settings:
   - `HF_MISTRAL_API_KEY`
   - `HF_OPENAI_API_KEY`

The application uses Hugging Face Spaces' persistent storage (`/data` directory) to maintain feedback data between restarts.

## Usage

1. Enter a factual question or select from example queries
2. Click "Detect Hallucinations" to start the analysis
3. Review the detailed results:
   - Hallucination detection status
   - Confidence score
   - Original and paraphrased responses
   - Detailed reasoning and analysis
4. Provide feedback to help improve the system

## How It Works

1. **Query Processing**:
   - Your question is paraphrased multiple ways
   - Each version is sent to Mistral Large
   - Responses are collected and compared

2. **Hallucination Detection**:
   - OpenAI's o3-mini analyzes responses
   - Identifies factual inconsistencies
   - Provides confidence scores and reasoning

3. **Feedback Collection**:
   - User feedback is stored in SQLite database
   - Persistent storage ensures data survival
   - Statistics are updated in real-time

## Data Persistence

The application uses SQLite for data storage in Hugging Face Spaces' persistent `/data` directory. This ensures:
- Feedback data survives Space restarts
- Statistics are preserved long-term
- No data loss during inactivity periods

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## About

This application uses a combination of paraphrasing techniques and model-as-judge approaches to identify potential hallucinations in LLM responses. It provides confidence scores, identifies conflicting facts, and offers detailed reasoning for its judgments.

## Features

- Generates paraphrased versions of input queries
- Evaluates responses using semantic similarity analysis
- Provides match percentage and similarity metrics
- Includes visualization tools for similarity matrices
- Web interface for interactive testing
- Benchmarking capabilities for bulk evaluation

## Installation

```bash
git clone https://github.com/serhanylmz/pas2
cd pas2
pip install -r requirements.txt
```

Set up your OpenAI API key in a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Web Interface

Run the Gradio interface:
```bash
python pas2-gradio.py
```

### Benchmark Tool

Run the benchmark tool:
```bash
python pas2-benchmark.py --json_file your_data.json --num_samples 10
```

### Library Usage

```python
from pas2 import PAS2

detector = PAS2()
hallucinated, response, questions, answers = detector.detect_hallucination(
    "your question",
    n_paraphrases=5,
    similarity_threshold=0.9,
    match_percentage_threshold=0.7
)
```

## Configuration

- Default model: gpt-4-2024-08-06
- Default embedding model: text-embedding-3-small
- Adjustable similarity and match percentage thresholds

## Output Files

- Similarity matrix plots (PNG)
- Match matrix plots (PNG)
- Benchmark results (CSV, TXT)
- User feedback logs (XLSX)

## License

This project is licensed under the MIT License with an attribution requirement - see the [LICENSE](LICENSE) file for details.

### Citation

If you use PAS2 in your research or project, please cite it as:

```bibtex
@software{pas2_2024,
  author = {Serhan Yilmaz},
  title = {PAS2 - Paraphrase-based AI System for Semantic Similarity},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/serhanylmz/pas2}
}
```

### Attribution Requirements

When using PAS2, you must provide appropriate attribution by:

1. Including the copyright notice and license in any copy or substantial portion of the software
2. Citing the project in any publications, presentations, or documentation that uses or builds upon this work
3. Maintaining a link to the original repository in any forks or derivative works

## Contact

Serhan Yilmaz
serhan.yilmaz@sabanciuniv.edu
Sabanci University