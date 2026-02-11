import warnings
from allennlp.predictors.predictor import Predictor
#from transformers import logging as transformers_logging

warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used.*")
warnings.filterwarnings("ignore", message="Some weights of BertModel.*were not used.*")
warnings.filterwarnings("ignore", category=FutureWarning)

def test() -> None:
    predictor = Predictor.from_path("models/structured-prediction-srl-bert.2020.12.15.tar.gz",
        cuda_device=-1
    )
    
    sentence = "The ENTITY_3a75013a shall detect wheel slip and shall activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km/h ."
    print(f"\nAnalyzing: '{sentence}'")
    result = predictor.predict(sentence=sentence)
    
    # for verb in result["verbs"]:
    #     print(f"\nVerb: {verb['verb']}")
    #     print(f"Description: {verb['description']}")
    print(result)

if __name__ == "__main__":
    test()

'''
OUTPUT ->
Analyzing: 'The ABS_ECU is required to detect wheel slip and shall activate braking within ten milliseconds when VehicleSpeed exceeds 30 kilometers per hour.'

Verb: is
Description: The ABS_ECU [V: is] required to detect wheel slip and shall activate braking within ten milliseconds when VehicleSpeed exceeds 30 kilometers per hour .

Verb: required
Description: [ARG2: The ABS_ECU] is [V: required] [ARG1: to detect wheel slip] and shall activate braking within ten milliseconds when VehicleSpeed exceeds 30 kilometers per hour .

Verb: detect
Description: [ARG0: The ABS_ECU] is required to [V: detect] [ARG1: wheel slip] and shall activate braking within ten milliseconds when VehicleSpeed exceeds 30 kilometers per hour .

Verb: shall
Description: The ABS_ECU is required to detect wheel slip and [V: shall] activate braking within ten milliseconds when VehicleSpeed exceeds 30 kilometers per hour .

Verb: activate
Description: [ARG0: The ABS_ECU] is required to detect wheel slip and [ARGM-MOD: shall] [V: activate] [ARG1: braking] [ARGM-TMP: within ten milliseconds] [ARGM-TMP: when VehicleSpeed exceeds 30 kilometers per hour] .

Verb: exceeds
Description: The ABS_ECU is required to detect wheel slip and shall activate braking within ten milliseconds [ARGM-TMP: when] [ARG0: VehicleSpeed] [V: exceeds] [ARG1: 30 kilometers per hour] .
-------------------------------------------------------------------------------------------------------------------

Analyzing: 'The ENTITY_3a75013a shall detect wheel slip and shall activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km/h .'

Verb: shall
Description: The ENTITY_3a75013a [V: shall] detect wheel slip and shall activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km / h .

Verb: detect
Description: [ARG0: The ENTITY_3a75013a] [ARGM-MOD: shall] [V: detect] [ARG1: wheel slip] and shall activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km / h .

Verb: shall
Description: The ENTITY_3a75013a shall detect wheel slip and [V: shall] activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km / h .

Verb: activate
Description: [ARG0: The ENTITY_3a75013a] shall detect wheel slip and [ARGM-MOD: shall] [V: activate] [ARG1: braking] [ARGM-TMP: within 10 ms] [ARGM-TMP: when ENTITY_51ebb1ae exceeds 30 km / h] .

Verb: exceeds
Description: The ENTITY_3a75013a shall detect wheel slip and shall activate braking within 10 ms [ARGM-TMP: when] [ARG0: ENTITY_51ebb1ae] [V: exceeds] [ARG1: 30 km / h] .
----------------------------------------------------------------------------------------------------------------------

Analyzing: 'The ENTITY_3a75013a shall detect wheel slip and shall activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km/h .'
{
'verbs': [
    {
    'verb': 'shall', 
    'description': 'The ENTITY_3a75013a [V: shall] detect wheel slip and shall activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km / h .', 
    'tags': ['O', 'O', 'B-V', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    }, 
    {
    'verb': 'detect', 
    'description': '[ARG0: The ENTITY_3a75013a] [ARGM-MOD: shall] [V: detect] [ARG1: wheel slip] and shall activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km / h .', 
    'tags': ['B-ARG0', 'I-ARG0', 'B-ARGM-MOD', 'B-V', 'B-ARG1', 'I-ARG1', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    }, 
    {'
    verb': 'shall', 
    'description': 'The ENTITY_3a75013a shall detect wheel slip and [V: shall] activate braking within 10 ms when ENTITY_51ebb1ae exceeds 30 km / h .', 
    'tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-V', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    }, 
    {
    'verb': 'activate', '
    description': '[ARG0: The ENTITY_3a75013a] shall detect wheel slip and [ARGM-MOD: shall] [V: activate] [ARG1: braking] [ARGM-TMP: within 10 ms] [ARGM-TMP: when ENTITY_51ebb1ae exceeds 30 km / h] .', 
    'tags': ['B-ARG0', 'I-ARG0', 'O', 'O', 'O', 'O', 'O', 'B-ARGM-MOD', 'B-V', 'B-ARG1', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'B-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'I-ARGM-TMP', 'O']
    }, 
    {
    'verb': 'exceeds', 
    'description': 'The ENTITY_3a75013a shall detect wheel slip and shall activate braking within 10 ms [ARGM-TMP: when] [ARG0: ENTITY_51ebb1ae] [V: exceeds] [ARG1: 30 km / h] .', 
    'tags': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ARGM-TMP', 'B-ARG0', 'B-V', 'B-ARG1', 'I-ARG1', 'I-ARG1', 'I-ARG1', 'O']
    }
        ], 
    'words': ['The', 'ENTITY_3a75013a', 'shall', 'detect', 'wheel', 'slip', 'and', 'shall', 'activate', 'braking', 'within', '10', 'ms', 'when', 'ENTITY_51ebb1ae', 'exceeds', '30', 'km', '/', 'h', '.']
}

'''