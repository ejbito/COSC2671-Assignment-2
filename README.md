# COSC2671 Assignment 2 - Comparative Network and Sentiment Analysis Pipeline

This project performs a comparative sentiment and network analysis of YouTube fandom communities for **Project Sekai** and **BanG Dream / Bandori** using the YouTube Data API v3.

The analysis pipeline collects YouTube videos, comments, and replies, preprocesses the data, performs sentiment analysis, constructs reply interaction networks, computes graph metrics, and generates comparative tables and visualisations for the final report.

## Pipeline Overview

The implementation is split into five stages:

1. **Data Collection**: collect YouTube videos, comments, replies, author IDs, and reply relationships.
2. **Data Preprocessing**: flatten and clean raw JSON into analysis-ready CSV files.
3. **Sentiment Analysis**: apply either VADER or RoBERTa sentiment analysis to cleaned comments.
4. **Network Analysis**: construct directed reply networks, detect communities, and compute network metrics.
5. **Combined Analysis and Visualisation**: merge sentiment and network outputs into final comparative tables and figures.

## Project Structure

```text
analysis/
|-- 01_preprocess.py
|-- 02_sentiment_analysis.py
|-- 03_network_analysis.py
`-- 04_combined_analysis.py

data/
|-- raw/
|-- vader/
|   |-- processed/
|   `-- outputs/
`-- roberta/
    |-- processed/
    `-- outputs/

src/
|-- __init__.py
|-- config.py
|-- fetchYoutubeData.py
`-- youtubeClient.py

pipeline.py
README.md
requirements.txt
```

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the NLTK VADER lexicon once:

```bash
python -m nltk.downloader vader_lexicon
```

RoBERTa requires `torch` and `transformers`. The first RoBERTa run downloads the model `cardiffnlp/twitter-roberta-base-sentiment-latest`, so it needs internet access and can be much slower than VADER.

On Windows, use Python 3.11 or 3.12 for RoBERTa if PyTorch fails to load under a newer Python version. A PyTorch DLL error such as `torch_python.dll` failing to load usually means the Python/PyTorch combination is not usable in that environment.

Sentiment labels are cached in the project-level cache directory:

```text
cache/vader_sentiment_cache.pkl
cache/roberta_sentiment_cache.pkl
```

RoBERTa writes to its cache after each batch, so rerunning the pipeline resumes from already labelled comments. If CUDA is available, the RoBERTa pipeline automatically uses GPU; otherwise it uses CPU.

## Environment Setup

Before collecting data, set your YouTube API key as an environment variable.

PowerShell:

```powershell
$env:YOUTUBE_API_KEY = "your_api_key_here"
```

Command Prompt:

```cmd
set YOUTUBE_API_KEY=your_api_key_here
```

macOS / Linux:

```bash
export YOUTUBE_API_KEY="your_api_key_here"
```

Do not include API keys, tokens, or credentials in submitted files.

## Run Order

Collect raw data:

```bash
python -m src.fetchYoutubeData
```

Run the analysis with VADER, which is the default:

```bash
python pipeline.py
```

Explicit VADER run:

```bash
python pipeline.py --vader
```

RoBERTa run:

```bash
python pipeline.py --roberta
```

The selected sentiment method controls the whole output directory:

```text
python pipeline.py --vader    -> data/vader/
python pipeline.py --roberta  -> data/roberta/
```

Raw data stays shared in `data/raw/`.

Compare VADER and RoBERTa after both pipelines have completed:

```bash
python analysis/sentiment_comparison.py
```

This writes comparison outputs to:

```text
data/sentiment_comparison/
```

## Expected Inputs

Raw data should be stored in:

- `data/raw/project_sekai_raw.json`
- `data/raw/bandori_raw.json`

## Expected Outputs

For VADER:

- `data/vader/processed/combined_comments_clean.csv`
- `data/vader/processed/combined_edges.csv`
- `data/vader/processed/combined_comments_with_network.csv`
- `data/vader/outputs/tables/*.csv`
- `data/vader/outputs/figures/*.png`
- `data/vader/outputs/graphs/*.gexf`

For RoBERTa:

- `data/roberta/processed/combined_comments_clean.csv`
- `data/roberta/processed/combined_edges.csv`
- `data/roberta/processed/combined_comments_with_network.csv`
- `data/roberta/outputs/tables/*.csv`
- `data/roberta/outputs/figures/*.png`
- `data/roberta/outputs/graphs/*.gexf`

For VADER vs RoBERTa comparison:

- `data/sentiment_comparison/tables/*.csv`
- `data/sentiment_comparison/figures/*.png`

## Network Construction Notes

- **Node:** unique `authorChannelId`
- **Edge:** a reply from user A to user B
- **Direction:** A -> B means user A replied to user B
- **Weight:** number of repeated replies between the same pair of users

The reply network is directed and weighted.

## Reproducibility Notes

- YouTube API quota limits may affect data collection.
- Running the collector multiple times may return slightly different videos depending on search ranking.
- Search results depend on the configured query lists.
- VADER and RoBERTa outputs are written separately so they can be compared in the report.
- Analysis scripts assume raw data has been collected successfully before execution.

## Dependencies

Required packages:

- google-api-python-client
- matplotlib
- networkx
- nltk
- numpy
- pandas
- scikit-learn
- torch
- tqdm
- transformers

## Submission Notes

- API keys are not included.
- Full datasets may not be included due to size limits.
- Representative sample data may be provided separately.
- Figures and tables generated by the pipeline correspond to outputs used in the final report.
