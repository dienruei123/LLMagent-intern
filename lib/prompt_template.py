SYSTEM_PROMPT = "You are a helpful assistant that can determine factuality of a statement. If the statement is telling the truth, output 'true'; otherwise, output 'false'."
USER_PROMPT = "Statement: {0}\nIs the statement telling the truth? Output only 'true' or 'false'. No need to explain.\n"
USER_PROMPT_Q = "Question: {0}\nIs the question telling the truth? Output only 'true' or 'false'. No need to explain.\n"
USER_PROMPT_Q_PASSAGE = "Question: {0}\n Reference Passage:\n{1}\n\nIs the question telling the truth based on reference passage? Output only 'true' or 'false'. No need to explain.\n"
USER_PROMPT_SCORE = "Statement: {0}\nIs the statement telling the truth? Please output true or false.\n\
Also, please output a confidence score of your answer, ranging from 0 to 10 (an integer).\n\
Higher scores indicates LLM is not lying, while lower scores indicates LLM may be hallucinating.\n\
Please output strictly according to the following format. DO NOT OUTPUT other irrevelant words. DO NOT EXPLAIN your answer.\n\
Format:\n\n\
Result: <your result, true or false>\n\
Score: <your confidence score, integer>\n\n\
"

