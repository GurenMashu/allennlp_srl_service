# A flask app served locally with waitress.
Has Predict and Batch Predict endpoints for pretrained semantic role labelling models from allennlp.

## Main issues i had with using allennlp:

- 1. Dependency issues. I have pip freezed my venv in requirements.txt but for the numpy ABI mismatch error that i encountered multiple times while installing allennlp and allennlp-models, refer to requirements.txt for a simple fix.

- 2. Use linux itself as windows is hell for such scenarios.

Other option is using allennlp's docker image.

## Usage

- Download the model from https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz and then replace the model path (in srl_server.py) with where you saved this.

- Then: 
```bash
pip install -r requirements.txt
python srl_server.py
```