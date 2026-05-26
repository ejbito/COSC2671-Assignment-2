# COSC2671 Assignment 2 - YouTube Network and Sentiment Analysis

This project compares two major rhythm game communities on YouTube:

- Project Sekai
- BanG Dream / Bandori

The analysis combines comment sentiment analysis with reply-network analysis. The main research question is:

```text
How do interaction network structures and sentiment patterns differ between two major rhythm game communities on YouTube?
```

## Python Version

Use **Python 3.12** for this project.

Python 3.14 may run the VADER pipeline, but RoBERTa depends on PyTorch, and PyTorch can fail to load native DLLs under unsupported or newer Python versions. The tested/recommended setup is a Python 3.12 virtual environment.

## Project Structure

```text
analysis/
|-- 01_preprocess.py
|-- 02_sentiment_analysis.py
|-- 03_network_analysis.py
|-- 04_combined_analysis.py
`-- 05_sentiment_comparison.py

cache/
|-- vader_sentiment_cache.pkl
`-- roberta_sentiment_cache.pkl

data/
|-- raw/
|-- vader/
|   |-- processed/
|   `-- outputs/
|-- roberta/
|   |-- processed/
|   `-- outputs/
`-- sentiment_comparison/
    |-- tables/
    `-- figures/

src/
|-- __init__.py
|-- config.py
|-- fetchYoutubeData.py
`-- youtubeClient.py

utils/
`-- create_sample_data.py

pipeline.py
requirements.txt
README.md
```

## Environment Setup

Create a Python 3.12 virtual environment:

```powershell
py -3.12 -m venv venv
```

If PowerShell blocks activation, allow scripts for the current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Activate the environment:

```powershell
.\venv\Scripts\Activate.ps1
```

Check the Python version:

```powershell
python --version
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader vader_lexicon
```

The included `requirements.txt` installs CUDA-enabled PyTorch wheels for NVIDIA GPU acceleration. This is recommended for the RoBERTa pipeline.

If PyTorch fails with a `torch_python.dll` error and you do not need GPU support, reinstall the CPU wheel:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(torch.__version__)"
```

## YouTube API Key

Before collecting data, set your YouTube API key as an environment variable.

PowerShell:

```powershell
$env:YOUTUBE_API_KEY = "your_api_key_here"
```

Do not include API keys, access tokens, or credentials in submitted files.

## Data Collection

Collect raw YouTube data:

```powershell
python -m src.fetchYoutubeData
```

This creates:

```text
data/raw/project_sekai_raw.json
data/raw/bandori_raw.json
```

You do not need to recollect data when switching between VADER and RoBERTa. Both sentiment methods use the same raw data.

## Run Analysis

Default run uses VADER:

```powershell
python pipeline.py
```

Explicit VADER run:

```powershell
python pipeline.py --vader
```

RoBERTa run:

```powershell
python pipeline.py --roberta
```

Faster RoBERTa run with larger batches and shorter token truncation:

```powershell
python pipeline.py --roberta --roberta-batch-size 64 --roberta-max-length 256
```

If your machine has enough RAM/VRAM, try `--roberta-batch-size 128`. If it crashes or slows down, use `64` or `32`.

Output directories are separated by sentiment method:

```text
python pipeline.py --vader    -> data/vader/
python pipeline.py --roberta  -> data/roberta/
```

## Sentiment Caching

Sentiment model outputs are cached as pickle files:

```text
cache/vader_sentiment_cache.pkl
cache/roberta_sentiment_cache.pkl
```

The cache is used only by `analysis/02_sentiment_analysis.py`. Later scripts read the scored CSV outputs from `data/vader/processed/` or `data/roberta/processed/`.

RoBERTa writes to the cache after each batch. If a run is interrupted, rerun:

```powershell
python pipeline.py --roberta
```

Already labelled comments will be skipped.

If CUDA is available, RoBERTa automatically uses GPU. Otherwise it uses CPU.

## Compare VADER and RoBERTa

After both pipelines complete:

```powershell
python analysis/05_sentiment_comparison.py
```

This reads:

```text
data/vader/processed/combined_comments_with_network.csv
data/roberta/processed/combined_comments_with_network.csv
```

and writes:

```text
data/sentiment_comparison/tables/
data/sentiment_comparison/figures/
```

The comparison script produces distribution plots, model agreement tables, a VADER vs RoBERTa heatmap, monthly and quarterly trend lines, and disagreement examples.

## Create Assignment Sample Data

The assignment asks for a representative sample of the data, no more than 10 MB, sufficient to show the structure used for network and NLP/text analysis.

After running the analysis, create the sample package:

```powershell
python utils/create_sample_data.py
```

This writes:

```text
sample_data/
```

The sample package includes compact raw JSON samples, processed comment samples, edge-list samples, summary tables, a manifest, and a README. It is intentionally small and does not include caches, API keys, credentials, or full datasets.

## Main Outputs

For each method:

```text
data/<method>/processed/combined_comments_clean.csv
data/<method>/processed/combined_edges.csv
data/<method>/processed/combined_comments_with_network.csv
data/<method>/outputs/tables/
data/<method>/outputs/figures/
data/<method>/outputs/graphs/
```

where `<method>` is either:

```text
vader
roberta
```

## Network Construction

- **Node:** YouTube user, represented by `authorChannelId`
- **Edge:** reply author -> parent comment author
- **Direction:** A -> B means user A replied to user B
- **Weight:** number of replies from A to B

The reply network is directed and weighted. Community detection and connected-component analysis use an undirected version of the reply network.
