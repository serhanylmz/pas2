# PAS2 - Paraphrase-based AI System for Semantic Similarity

A hallucination detection system that uses paraphrasing and semantic similarity analysis to evaluate the consistency of AI responses.

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