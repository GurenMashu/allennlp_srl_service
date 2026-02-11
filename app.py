from fastapi import FastAPI
from pydantic import BaseModel
from allennlp.predictors.predictor import Predictor

app = FastAPI(title = "Allennlp SRL Service")

class SRLRequest(BaseModel):
    sentence: str

class SRLResponse(BaseModel):
    verbs: list

predictor = Predictor.from_path(r"allennlp_srl_service/models/srl-bert-2020.12.15.tar.gz")

@app.post("/srl", response_model = SRLResponse)
def srl(req: SRLRequest):
    output = predictor.predict(sentence = req.sentence)
    return {"verb": output["verbs"]}
