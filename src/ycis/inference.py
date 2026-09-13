# bert_predictor.py # sentiment analysis by fine-tuned BERT transformer

import torch
import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from ycis.data.fetchData import get_comments
from ycis.data.preprocess import filter_english_comments
from huggingface_hub import snapshot_download

from ycis.config import Config
config = Config()

labels = {0: "negative", 1: "positive"}


class Predictor:
    def __init__(self, model_path, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if self.device == "cuda":
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path
            ).to(self.device)
        else:
            self.model = ORTModelForSequenceClassification.from_pretrained(
                model_path,
                export=False,
                provider="CPUExecutionProvider",  # ONNX form
            )

    def predict_text(self, text):
        if self.device.type == "cuda":
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred = torch.argmax(probs).item()

            return {
                "text": text,
                "prediction": pred,
                "confidence": float(torch.max(probs)),
            }

        else:
            inputs = self.tokenizer(text, return_tensors="pt")
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            if hasattr(logits, "detach"):
                logits = logits.detach().cpu().numpy()

            exp_logits = np.exp(
                logits - np.max(logits, axis=-1, keepdims=True)
            )  # Softmax (numpy version)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            pred = int(np.argmax(probs))
            confidence = float(np.max(probs))

            return {"text": text, "prediction": pred, "confidence": confidence}

    def predict_df(self, df, text_column="text", batch_size=config.BATCH_SIZE, max_length=config.MAX_LENGTH):
        texts = df[text_column].fillna("").astype(str).tolist()
        print(f"Number of comments: {len(texts)}")

        all_labels = []
        all_scores_0 = []
        all_scores_1 = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = self.tokenizer(
                batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            ).to(self.device)

            if self.device.type == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            else:
                outputs = self.model(**inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                if hasattr(logits, "detach"):
                    logits = logits.detach().cpu().numpy()
                
                exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            labels = np.argmax(probs, axis=1).tolist()
            all_labels.extend(labels)
            all_scores_0.extend(probs[:, 0].tolist())
            all_scores_1.extend(probs[:, 1].tolist())

        df["bert_preds"] = all_labels 
        df["scores_0"] = all_scores_0
        df["scores_1"] = all_scores_1

        return df

if __name__ == "__main__":
    bert_path = snapshot_download(repo_id=config.TRAINED_MODEL_ID)
    bert = Predictor(bert_path)
    result = bert.predict_text("I’m super frustrated by it.")
    print(f"Text: '{result['text']}'| prediction: {labels[result['prediction']]} | confidence: {result['confidence']}")

    video_id = "SbNDmAJBtyU" # example vdieo id
    comments_df = get_comments(video_id, 30, 1)
    english_comments_df = filter_english_comments(comments_df)
    results = bert.predict_df(english_comments_df)
    print(results.to_csv(config.DATA_DIR / 'comments_sample.csv', index=False))