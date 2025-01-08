# LLMagent-intern

### Abstract
To be updated

## Experiment Results
Refer to [Zresults.md](./Zresults.md)

## Data preprocessing

**Remember to place huggingface key in .env!!**

### Generate training dataset from *Internal of LLM*
```bash
python -m preprocess.truefalse_dataset

# Args

# --layer {0~31} Specify the order of layer (0-indexed)
# --dataset {} Specify dataset name
# --dest {train, test} Specify destination directory for each train, test data (default: test)

# Example: Layer 19, animal dataset
python -m preprocess.truefalse_dataset --layer 19 --dataset animals --dest train
```

#### Options for Internal of LLM train dataset names
{animals, cities, companies, elements, facts, inventions}
```bash
python -m preprocess.truefalse_dataset --layer 19 --dataset cities --dest train
```

#### Options for Internal of LLM test dataset names
{generated}
```bash
python -m preprocess.truefalse_dataset --layer 19

# Equivalent to: 
python -m preprocess.truefalse_dataset --layer 19 --dataset generated --desc test
```
Note that `generated` dataset can only be generated for the purpose of testing dataset.

#### All dataset
If you want to generate all datasets from Internal of LLM, you should first generate datasets for all six categories first. Then specify `--dataset all` to concat datasets.

```bash
python -m preprocess.truefalse_dataset --layer <your target layer> --dataset all
```

### Generate training dataset from *Boolq*
```bash
python -m preprocess.boolq

# Args

# --layer {0~31} Specify the order of layer (0-indexed)
# --ref   Specify whether referenced text is added

# Will generate both train and test data

# Example: layer 19, add reference text
python -m preprocess.boolq --layer 19 --ref

# Example: layer 25, no reference text
python -m preprocess.boolq --layer 25
```

## Training Models

### Training a Classifier
In `model_train.py`

```bash
python -m model_train

# Args
# --layer {0~31} Specify the order of layer (0-indexed)
# --dataset {} Specify dataset name
# --model_name  {(str)} Specify the name of the classifier
# --ref   Specify whether referenced text is added (only for boolq)
# --equal   Specify whether equal-labeled dataset is used


# Example: Training layer 19, animal dataset, no equal label
# Model will be saved to "models/animals_layer19.ckpt"
python -m model_train --layer 19 --dataset animals --model_name animals_layer19

# Example: Training layer 27, boolq dataset, referenced text, equal labeled
python -m model_train --layer 27 --dataset boolq --model_name boolq_ref_equal_layer27 --ref --equal
```

### Training a Domain Adaptation classifier
In `domain_adaptation.py`

- *Note: There are two versions of DA.*
- *Note: DA Classifiers are only trained on Internal State of LLM Datasets.*

```bash
# Version 1
python -m domain_adaptation

# Version 2
python -m domain_adaptation_oneepoch


# Args
# --layer {0~31} Specify the order of layer (0-indexed)
# --lamb {0.005, 0.02, 0.1, 0.2, 0.3, 0.5} Specify lambda value (regularization of D loss)
# --equal   Specify whether equal-labeled dataset is used


# Example: Version 1, Training layer 19, lambda=0.1, no equal label
# Saved model has prefix "extractor", "predictor" and suffix ".bin"
python -m domain_adaptation --layer 19 --lamb 0.1
```

Detailed implementation for different versions will be introduced soon. Stay tuned!

## Testing Model

### Test a single model

In `model_testdataset.py`


```bash
python -m model_testdataset

# Args
# --layer {0~31} Specify the order of layer (0-indexed)
# --dataset {} Specify dataset name
# --model_name  {(str)} Specify the name of the classifier
# --ref   Specify whether referenced text is added (only for boolq)


# Example: Testing layer 19, trained on animal dataset, no equal label
# Note: generated is the testing set for Internal State of LLM
python -m model_testdataset --layer 19 --dataset generated --model_name animals_layer19


# Example: Testing layer 27, trained on boolq dataset, equal labeled, referenced text
python -m model_testdataset --layer 27 --dataset boolq --model_name boolq_ref_equal_layer27 --ref
```


Output result format:
```txt
[Testing: (<test dataset name>, Layer <# layer>)]
Test loss: 
Test acc: %
[True stats]
F1 Score: (highest 100)
Recall: 
Precision: 
[False stats]
F1 Score: (highest 100)
Recall: 
Precision: 
```

### Voting (multiple classifiers)
In `model_test_vote.py`

```bash
python -m model_test_vote

# Manual modifications so far
```

### Cross testing
Use `model_testdataset.py`, replace the dataset to different one.

```bash
# Example: Testing layer 19, trained on animal dataset, testing on boolq
python -m model_testdataset --layer 19 --dataset boolq --model_name animals_layer19


# Example: Testing layer 27, trained on boolq dataset, equal labeled, referenced text, testing on all (Internal State of LLM) dataset
python -m model_testdataset --layer 27 --dataset all --model_name boolq_ref_equal_layer27
```

### Cross testing - Domain Adaptation
In `domain_adaptation_test.py`

```bash
python -m domain_adaptation_test


# Args
# --layer {0~31} Specify the order of layer (0-indexed)
# --lamb {0.005, 0.02, 0.1, 0.2, 0.3, 0.5} Specify lambda value (regularization of D loss)
# --ref   Specify whether referenced text is added (only for boolq)

# Manual Modification part:
# Uncomment the model you want to test (no args to specify model name at prompt stage)
```

## Notes
For other Details, you could refer to `script.py` for more bash prompt usages.

Meeting discussions: https://www.notion.so/6e01335cd6ab4f6481bd70de406565c6
