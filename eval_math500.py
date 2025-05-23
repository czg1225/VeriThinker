from dataset import extract_all_boxed_content
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
from datasets import load_dataset
from dataset import extract_answer_llm
from vllm import LLM, SamplingParams
import re
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig


def main():
    test_examples = load_dataset("HuggingFaceH4/MATH-500",split="test")
    test_examples = list(test_examples)*1

    seed = 42
    
    model_name = "Zigeng/R1-VeriThinker-7B"

    llm = LLM(model=model_name,tensor_parallel_size=4, max_model_len=40000)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    right = 0
    tokens = 0
    answers = []
    
    
    # Define batch size
    batch_size = 500
    
    for i in range(0, len(test_examples), batch_size):
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=16384, seed=seed+i, stop=["\n</think>"])

        end = min(i + batch_size, len(test_examples))  
        batch_examples = test_examples[i:end]

        batch_prompts = []
        batch_gt_answers = []
        
        # Prepare prompts for the batch
        for example in batch_examples:
            prompt = example["problem"]

            # deepseek r1
            tail = r" Please reason step by step, and put your final answer within \boxed{}."

            messages = [
            {"role": "user", "content": prompt + tail}
            ]


            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )


            batch_prompts.append(text)
            batch_gt_answers.append(example["answer"])
        
        
        # Generate responses for batch
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process each response in the batch
        for j, (output) in enumerate(outputs):

            response = output.outputs[0].text

            num_new_tokens = len(tokenizer.encode(response))
            
            gt_answer = batch_gt_answers[j]

######################################################################
            solution = response
            expected_answer = gt_answer

            # Extract prediction wrapped by "\\boxed{}"
            prediction_match = extract_all_boxed_content(str(solution))
            if len(prediction_match) > 0:
                prediction = prediction_match[-1]
                if prediction is not None and '\\boxed' in prediction:
                    prediction = prediction.replace('\\boxed{', '')[:-1]
            else:
                patterns = [
                    r"<answer>(.*?)</answer>",
                    r"</answer>(.*?)</answer>",
                    r"<answer>(.*?)<answer>",
                    r"\*\*Answer:\*\* ([\d\.]+)",
                ]
                for pattern in patterns:
                    prediction_match = re.findall(pattern, str(solution))
                    if len(prediction_match) > 0:
                        break
                    
                if len(prediction_match) > 0:
                    prediction = prediction_match[-1]
                else:
                    prediction = None

            # Check if prediction matches the expected answer
            if prediction is not None:#prediction == expected_answer:
                gold = parse("$"+expected_answer+"$", extraction_config=[LatexExtractionConfig()])
                answer = parse("$"+prediction+"$", extraction_config=[LatexExtractionConfig()])
                if verify(gold, answer):
                # if grade_answer(prediction, expected_answer):
                    right += 1
                    accept = True
                else:
                    pure_number_prediction = re.findall(r"[-+]?\d*\.\d+|\d+", prediction)
                    pure_number_expected_answer = re.findall(r"[-+]?\d*\.\d+|\d+", expected_answer)
                    if pure_number_prediction and pure_number_expected_answer and float(pure_number_prediction[0]) == float(pure_number_expected_answer[0]):
                        right += 1
                        accept = True
                    else:
                        accept = False
            else:
                accept = False

            if prediction is None:
                prediction = extract_answer_llm(response)
                gold = parse("$"+expected_answer+"$", extraction_config=[LatexExtractionConfig()])
                answer = parse("$"+prediction+"$", extraction_config=[LatexExtractionConfig()])
                if verify(gold, answer):
                    right += 1
                    accept = True
            llm_answer = prediction
            
            answers.append({
                "question": batch_examples[j]["problem"],
                "gt_answer": gt_answer,
                "llm_answer": llm_answer,
                "accept":accept,
                "llm_response": response,
                "tokens": num_new_tokens,
            })
            

            # Update token count with actual new tokens
            tokens += num_new_tokens
            
            # Print progress for each example
            current_idx = i + j + 1
            print("sample num:", current_idx, "result:", accept, "accuracy:", right/current_idx)
            print("gt_answer:", gt_answer, "llm_answer:", llm_answer)
            print('tokens:', num_new_tokens)
            print('-' * 50)
        
    
    
    # Calculate final metrics
    avg_tokens = tokens / len(test_examples)
    ratio = right / len(test_examples)
    
    print("#############################################MATH500#############################################")
    print("num of samples:", len(test_examples))
    print("avg tokens:", avg_tokens)
    print("avg accuracy:", ratio)
    
    # Save wrong answers to JSON
    with open("test_math500.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()