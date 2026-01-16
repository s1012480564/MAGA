from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import uvicorn

app = FastAPI(title="Reward Model API")

roberta_path = ""

roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_path)
reward_model = RobertaForSequenceClassification.from_pretrained(roberta_path)

reward_model = reward_model.to(0)
reward_model.eval()


class ScoreRequest(BaseModel):
    solution_str: str


class ScoreResponse(BaseModel):
    reward_score: float


@app.post("/compute_score", response_model=ScoreResponse)
def api_compute_score(request: ScoreRequest):
    inputs = roberta_tokenizer(request.solution_str, padding=True, truncation=True, return_tensors="pt").to(0)

    with torch.no_grad():
        outputs = reward_model(**inputs)

    probs = outputs.logits.softmax(dim=-1)
    human_prob = probs[0, 0]
    reward_score = float(human_prob)

    return {"reward_score": reward_score}


if __name__ == "__main__":
    uvicorn.run(
        app="reward_server:app",
        host="0.0.0.0",
        port=38000,
        reload=False
    )