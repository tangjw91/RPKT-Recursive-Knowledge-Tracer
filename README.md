# RPKT: Recursive Prerequisite Knowledge Tracer

An AI-powered educational tool that helps learners discover knowledge gaps by recursively tracing prerequisite concepts using GPT-4o.

For full details on the system design, methodology, and evaluation, please refer to our paper:

> **J. Tang, Q. Guo, Z. Tang and Y. Shang**, "RPKT: Learning What You Don't Know – Recursive Prerequisite Knowledge Tracing in Conversational AI Tutors for Personalized Learning," *2025 IEEE International Conference on Future Machine Learning and Data Science (FMLDS)*, pp. 133-138, 2025.
>
> DOI: [10.1109/FMLDS67896.2025.00031](https://ieeexplore.ieee.org/document/11446819)

## Getting Started

### Prerequisites

- Python 3.9+
- An [OpenAI API key](https://platform.openai.com/api-keys) with GPT-4o access

### Installation

1. Clone the repository:

```bash
git clone https://github.com/tangjw91/RPKT-Recursive-Knowledge-Tracer.git
cd RPKT-Recursive-Knowledge-Tracer
```

2. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your API key:

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

5. Run the app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Citation

If you use this work, please cite:

```bibtex
@INPROCEEDINGS{11446819,
  author={Tang, Jinwen and Guo, Qiming and Tang, Zhicheng and Shang, Yi},
  booktitle={2025 IEEE International Conference on Future Machine Learning and Data Science (FMLDS)},
  title={RPKT: Learning What You Don't Know – Recursive Prerequisite Knowledge Tracing in Conversational AI Tutors for Personalized Learning},
  year={2025},
  pages={133-138},
  doi={10.1109/FMLDS67896.2025.00031}
}
```
