import torch

from lib.LLM import LLM
from lib.classifier import TrueFalseClassifier

def test_truth(llm: LLM, model, prompt):
    system_message = "You are a helpful assistant that can determine factuality of a statement. If the statement is telling the truth, output 'true'; otherwise, output 'false'."
    user_message = "Statement: " + prompt + "\nIs the statement telling the truth? Output only 'true' or 'false'. No need to explain.\n"
    refined_input = f"[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{user_message}[/INST]"
    
    state, llm_output = llm.get_llm_state_and_output(refined_input)
    with torch.no_grad():
        outputs = model(state)
    # print(outputs)
    pred = torch.argmax(outputs)
    return llm_output, True if pred == 1 else False
    
device = 'cuda' if torch.cuda.is_available else 'cpu'
# test_prompt = "Sun rises in the east." # True
# test_prompt = "Indium is in the Lanthanide group." # False
dataset_llm = LLM(device)

model = TrueFalseClassifier()
model.to(device)
model.load_state_dict(torch.load('./models/model.ckpt', weights_only=True))
model.eval()

print("********* Start testing **********")

while True:
    test_prompt = input("Input prompt (type 'quit' to exit): ")
    if 'quit' in test_prompt:
        break
    
    # test_prompt = "Papé is a type or brand of paper."
    llm_result, classifier_result = test_truth(dataset_llm, model, test_prompt)
    print("******** Result ********")
    print("** LLM Result **: ", llm_result)
    print("** Classifier Result **: ", classifier_result)

print("******** Finished testing ********")