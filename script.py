import os
# from lib.args import dataset_choices
dataset_choices = [
    'animals',
    'cities',
    'companies',
    'elements',
    'facts',
    'inventions',
]

# # Voting - Single
# for layer in range(15, 28, 4):
#     os.system(f'python model_train.py --layer {layer} --dataset all --model_name single_all_{layer}')
#     os.system(f'python model_train.py --layer {layer} --dataset all --model_name single_all_equal_{layer} --equal')

# # Boolq - Single
# for layer in range(15, 28, 4):
#     os.system(f'python model_train.py --layer {layer} --dataset boolq --model_name single_boolq_{layer}')
#     os.system(f'python model_train.py --layer {layer} --dataset boolq --model_name single_boolq_equal_{layer} --equal')
    
#     os.system(f'python model_train.py --layer {layer} --dataset boolq --model_name single_boolq_ref_{layer} --ref')
#     os.system(f'python model_train.py --layer {layer} --dataset boolq --model_name single_boolq_ref_equal_{layer} --ref --equal')

# os.system(f'python model_train.py --layer {15} --dataset boolq --model_name single_boolq_ref_{15}.ckpt --ref')

# for layer in range(15, 28, 4):
#     os.system(f'python model_train.py --layer {layer} --dataset boolq --model_name single_boolq_ref_equal_{layer} --ref --equal')

    
    
# for dataset_name in dataset_choices:
#     if dataset_name in ['generated', 'all']:
#         continue
#     for layer in range(3, 15, 2):
#         os.system(f'python truefalse_dataset.py --layer {layer} --dest train --dataset {dataset_name}')

# for layer in range(3, 15, 2):
#     os.system(f'python truefalse_dataset.py --layer {layer}')

# for layer in range(15, 32, 4):
#     os.system(f'python boolq.py --layer {layer}')

# os.system(f"python boolq.py --layer 19")
# for layer in range(23, 32, 4):
#     os.system(f"python boolq.py --ref True --layer {layer}")
    

# for layer in range(15, 28, 4):
#     # test all(internal state) classifier on BoolQ dataset
#     os.system(f'python model_testdataset.py --layer {layer} --dataset boolq --model_name model_all_{layer}')
    

# for category in dataset_choices:
#     for layer in range(15, 28, 4):
#         # os.system(f'python model_train.py --layer {layer} --dataset {category} --model_name model_{category}_{layer}')
#         os.system(f'python model_testdataset.py --layer {layer} --dataset boolq --model_name model_{category}_{layer}')

# for layer in range(15, 28, 12):
#     # test all(internal state) classifier on BoolQ dataset
#     os.system(f'python model_testdataset.py --layer {layer} --dataset generated --model_name model_boolq_{layer}')

# for layer in range(15, 28, 12):
#     print(f"Layer {layer} training:")
#     os.system(f"python model_train.py --dataset boolq --layer {layer} --model_name model_boolq_{layer}")
    # os.system(f"python model_train.py --dataset boolq --ref True --layer {layer} --model_name model_boolq_passage_{layer}")\
        
# for layer in range(15, 28, 12):
#     print(f"Layer {layer} training:")
#     os.system(f"python model_train.py --dataset boolq --ref True --layer {layer} --model_name model_boolq_passage_{layer}")

for lamb in [0.005, 0.02, 0.1, 0.2, 0.3, 0.5]:
    print("Lambda = {} DA (Equal) training...".format(lamb))
    os.system(f"python domain_adaptation.py --layer 19 --equal --lamb {lamb}")
    
# for lamb in [0.005, 0.02, 0.1, 0.2, 0.3, 0.5]:
#     print("Lambda = {} DA_One training...".format(lamb))
#     os.system(f"python domain_adaptation_oneepoch.py --layer 19 --lamb {lamb}")

# Testing
# Go to domain_adaptation_test.py to change mode
# for lamb in [0.005, 0.02, 0.1, 0.2, 0.3, 0.5]:
#     print("Lambda = {} testing...".format(lamb))
#     os.system(f"python domain_adaptation_test.py --layer 19 --lamb {lamb}")

    

