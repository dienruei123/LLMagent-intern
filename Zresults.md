# Results

### Architecture

NN Part: `4096-512-512-256-64-2`
DaNN Domain: `512-512-512-512-512-6`

### Single

#### Dataset from Internal State of LLM

Test: `generated_true_false`
Train Epoch: 60

| Methods                            | Accuracy | Details                      |
| ---------------------------------- | -------- | ---------------------------- |
| Vanilla                            | 61.22 %  |
| Verbal                             | 66.94 %  | @2024/9/23                   |
| Classifier (previous)              | 76.73 %  | layer 20, @2024/9/18         |
| Classifier (new, original data)    | 78.37 %  | layer 24, same arch. as DaNN |
| Classifier (new, equal label data) | 74.69 %  | layer 20, same arch. as DaNN |
| Voting                             | 76.73 %  | layer 16, 20, 24; original   |

##### Voting Details

Train Layer 12, 16, 20, 24 (Pick 3 layers for each test), decide the best combination

#### Dataset from Boolq

Train Epoch: 60

| Methods                            | Accuracy | Details                                  |
| ---------------------------------- | -------- | ---------------------------------------- |
| Vanilla                            | 62.11 %  |
| Verbal                             | 60.61 %  | @2024/10/7 (no passage)                  |
| Classifier (previous)              | 66.82 %  | layer 24                                 |
| Classifier (new, original data)    | 70.86 %  | layer 20, same arch. as DaNN; + ref text |
| Classifier (new, equal label data) | 61.83 %  | layer 16, same arch. as DaNN; + ref text |
| Voting                             | 71.13 %  | layer 16, 20, 24; + ref text             |

### Cross Dataset Test

#### Internal State of LLM (all train data) → Boolq

| Methods                            | Accuracy | Details                      |
| ---------------------------------- | -------- | ---------------------------- |
| Classifier (previous)              | 51.87 %  | layer 24                     |
| Classifier (new, original data)    | 54.01 %  | layer 20, same arch. as DaNN |
| Classifier (new, equal label data) | 48.38 %  | layer 28, same arch. as DaNN |
| Voting                             | 52.81 %  | layer 20, 24, 28; original   |
| DaNN                               | 58.81 %  | layer 20, 100 epoch          |

#### Internal State of LLM (all train data) → Boolq + reference text

| Methods                            | Accuracy | Details                      |
| ---------------------------------- | -------- | ---------------------------- |
| Classifier (previous)              |
| Classifier (new, original data)    | 61.44 %  | layer 20, same arch. as DaNN |
| Classifier (new, equal label data) | 54.46 %  | layer 20, same arch. as DaNN |
| Voting                             | 60.89 %  | layer 16, 20, 28; original   |
| DaNN                               | 67.34 %  | layer 20, 20 epoch           |

##### Boolq → Internal State of LLM (not important)

| Methods                            | Accuracy | Details  |
| ---------------------------------- | -------- | -------- |
| Classifier (previous)              |
| Classifier (new, original data)    | 50.58 %  | layer 24 |
| Classifier (new, equal label data) | 50.58 %  | layer 20 |

### Other Details
