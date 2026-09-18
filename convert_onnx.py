from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from ycis.config import Config
config = Config()

model_id = config.TRAINED_MODEL_ID
output_dir =  config.PROJECT_ROOT / "onnx_models/bert-sentiment-model"

model = ORTModelForSequenceClassification.from_pretrained(
    model_id,
    export=True,
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"Saved ONNX model to: {output_dir}")