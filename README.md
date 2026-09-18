# YouTube Comment Intelligence System

Analyze English YouTube comments with BERT sentiment classification, sentiment-specific topic modeling, and LLM-generated topic summaries.

The project provides Gradio applications for local use, Azure Container Apps, and Hugging Face Spaces.

Live Demo through Azure: https://ycis-app.ashyfield-c8f8fb00.germanywestcentral.azurecontainerapps.io

## What It Does

For a YouTube video ID or URL, the application:

1. Fetches a user-specified number of comments through the YouTube Data API v3.
2. Removes URLs and keeps comments detected as English.
3. Classifies comments as `negative` (`0`) or `positive` (`1`) with a fine-tuned BERT model.
4. Generates `all-MiniLM-L6-v2` embeddings and stores comments, labels, and embeddings in SQLite.
5. Runs BERTopic with UMAP and HDBSCAN separately for each sentiment.
6. Sends comments from a selected sentiment/topic group to Groq for a concise summary.

The Gradio UI shows sentiment counts, positive and negative topic tables, and the comments used for each summary.

## Architecture

```text
YouTube Data API -> fetchData.py -> preprocess.py -> inference.py
                                                       |
                                                       v
                                        storage.py + SQLite embeddings
                                                       |
                                                       v
                                             topic_modeling.py
                                                       |
                                                       v
                                                   llm.py + Groq
                                                       |
                                                       v
                                                  Gradio UI
```

Important details:

- The default configuration fetches 100 comments per page and up to 5 pages. 
- The sentiment model is downloaded from `sweetguma/bert-sentiment-model` at startup. CPU inference uses the checked-in ONNX model.
- The default database is `data/DB/comments_cache.db`. Tables are named after the video ID, with hyphens replaced by underscores.
- Topic modeling skips a sentiment group with fewer than 10 comments. Topic `-1` is excluded from the topic tables displayed by the UI.

## Requirements

- Python 3.12 or newer
- `uv` for environment and dependency management
- YouTube Data API v3 key
- Groq API key for topic summaries
- Network access on first startup for Hugging Face and embedding model downloads

The project depends on PyTorch, ONNX Runtime, BERTopic, HDBSCAN, Sentence Transformers, Gradio, and LangChain integrations. The Azure image installs the required build tools (see pyproject.toml).

## Local Setup

```bash
uv sync
```

Create `.env` in the project root:

```dotenv
YOUTUBE_API_KEY=your_youtube_api_key
GROQ_API_KEY=your_groq_api_key
MAX_RESULTS=100
MAX_PAGES=5
```

`.env.example` contains the same variables. Keep API keys out of source control. The first import of `ycis.config` initializes the shared embedding model and downloads the sentiment model, so startup can take several minutes and use substantial memory.

## Run the Gradio App

Run the Azure-compatible app locally:

```bash
uv run python deployment/azure/app.py
```

It listens on `http://localhost:7860` by default. Use `PORT` to change the port:

```bash
python3 deployment/azure/app.py
```

The Hugging Face entry point can also be run locally:

```bash
uv run python deployment/huggingface/app.py
```

It uses port `8080` locally and platform defaults when `SPACE_ID` is present. Enter a video ID or a full YouTube URL. An empty input uses the fallback video ID built into the app.

## Programmatic Workflow

The core pipeline can be used without Gradio:

```python
from ycis.config import Config
from ycis.data.fetchData import get_comments
from ycis.data.preprocess import filter_english_comments
from ycis.data.storage import save_to_db_with_embedding
from ycis.inference import Predictor
from ycis.topic_modeling import update_topics_in_db
from ycis.llm import get_comments_from_db, summarize_topic_comments

config = Config()
video_id = "XqYTfpxFuDM"
db_name = "comments_cache.db"

comments = filter_english_comments(get_comments(video_id))
predictor = Predictor(config.TRAINED_MODEL_PATH)
comments = predictor.predict_df(comments)
save_to_db_with_embedding(video_id, comments, db_name)

db_path = config.DB_DIR / db_name
topic_info, topic_model = update_topics_in_db(
    db_path, video_id, sentiment_label=1
)
selected_comments = get_comments_from_db(
    db_path, video_id, sentiment_label=1, topic=0
)
print(summarize_topic_comments(selected_comments))
```

`Predictor.predict_text()` returns the text, numeric prediction, and confidence. `predict_df()` adds `bert_preds`, `scores_0`, and `scores_1` columns in place.

## Docker

```bash
docker build -f Dockerfile.azure -t ycis-azure .
docker run --rm -p 7860:7860 --env-file .env ycis-azure
```

`Dockerfile.azure` installs production dependencies with `uv sync --no-dev`, copies the source and ONNX model, and exposes port `7860`.

## Azure Container Apps

The deployment flow is:

1. Create an Azure resource group, Container Registry, and Container Apps environment.
2. Build, tag, and push the image to ACR.
3. Create the Container App with external ingress and target port `7860`.
4. Store `YOUTUBE_API_KEY` and `GROQ_API_KEY` as Container App secrets and expose them as environment variables.
5. Read the ingress FQDN and inspect logs with `az containerapp logs show`.

Example after the registry and environment exist:

```bash
docker build -f Dockerfile.azure -t ycis-azure .
docker tag ycis-azure <registry>.azurecr.io/ycis-azure:latest
docker push <registry>.azurecr.io/ycis-azure:latest

az containerapp create \
  --name ycis-app \
  --resource-group <resource-group> \
  --environment <container-apps-environment> \
  --image <registry>.azurecr.io/ycis-azure:latest \
  --target-port 7860 \
  --ingress external \
  --registry-server <registry>.azurecr.io \
  --registry-username <registry-username> \
  --registry-password "$ACR_PASSWORD"
```

For Consumption workloads, Azure requires compatible CPU and memory pairs. The documented `1.0` CPU and `2Gi` memory combination is valid for this workload.

## Hugging Face Spaces

Use `deployment/huggingface/app.py` as the Space application entry point. Configure `YOUTUBE_API_KEY` and `GROQ_API_KEY` as Space secrets. The application detects a Space through `SPACE_ID` and launches Gradio with platform configuration.

The repository does not include a Hugging Face-specific `requirements.txt`; install dependencies from `pyproject.toml` in the Space build configuration or use the supported `uv` workflow.

## Repository Layout

```text
src/ycis/
  config.py              Environment variables, model paths, shared embeddings
  inference.py           BERT/ONNX sentiment prediction
  llm.py                 SQLite retrieval and Groq summaries
  topic_modeling.py      UMAP, HDBSCAN, and BERTopic clustering
  data/fetchData.py      YouTube comment retrieval
  data/preprocess.py     URL removal and English filtering
  data/storage.py        Embedding generation and SQLite persistence
deployment/azure/app.py   Azure Gradio app
deployment/huggingface/app.py  Hugging Face Gradio app
onnx_models/              Local ONNX model and tokenizer
data/comments_sample.csv  Example processed output
data/DB/                  Runtime SQLite databases
convert_onnx.py           Export the sentiment model to ONNX
test.py                   Manual end-to-end workflow
test.ipynb                Exploratory notebook workflow
Dockerfile.azure          Azure container image
```

## Export the ONNX Model

Regenerate the checked-in ONNX model and tokenizer with:

```bash
uv run python convert_onnx.py
```

## Limitations

- Comments shorter than six characters are discarded, and only English comments are retained.
- Topic summaries use the full selected comment group and may be slow or exceed the LLM context limit for large groups.
- No automated test suite is configured.

<!-- ## License

No license file is currently included in the repository. -->
