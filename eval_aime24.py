from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import time
from datasets import load_dataset
from dataset import extract_answer_llm
from vllm import LLM, SamplingParams
from dataset import extract_answer_llm, extract_answer_qwq, is_float


def main():

    test_examples = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    test_examples = list(test_examples)
    test_examples = test_examples*10

    seed=42

    model_name = "model/R1_VeriThinker_Qwen_7B"

    llm = LLM(model=model_name,tensor_parallel_size=4, max_model_len=40000)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    right = 0
    tokens = 0
    answers = []
    
    start_time = time.time()
    
    # Define batch size
    batch_size = 30
    
    for i in range(0, len(test_examples), batch_size):
        sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=16384, seed=seed+i, stop=["\n</think>"])
        end = min(i + batch_size, len(test_examples))  
        batch_examples = test_examples[i:end]

        batch_prompts = []
        batch_gt_answers = []
        
        # Prepare prompts for the batch
        for example in batch_examples:
            prompt = example["Problem"]


            #deepseek r1
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
            batch_gt_answers.append(example["Answer"])
        
        
        # Generate responses for batch
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process each response in the batch
        for j, (output) in enumerate(outputs):

            response = output.outputs[0].text

            num_new_tokens = len(tokenizer.encode(response))
            
            gt_answer = batch_gt_answers[j]

######################################################################

            llm_answer = extract_answer_qwq(response)
            if is_float(llm_answer):
                llm_answer = llm_answer
            else:
                llm_answer = extract_answer_llm(response)
            print(gt_answer, "||", llm_answer)
            if is_float(gt_answer) and is_float(llm_answer):
                try:
                    accept = ( int(round(float(gt_answer)))==int(round(float(llm_answer))) )
                except OverflowError:
                    accept = False
            else:
                accept = False
            
            if accept:
                right += 1
            answers.append({
                "question": batch_examples[j]["Problem"],
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
            print("sample num:", current_idx, "result:", accept, "right ratio:", right/current_idx)
            print("gt_answer:", gt_answer, "llm_answer:", llm_answer)
            print('tokens:', num_new_tokens)
            print('-' * 50)
        
    
    end_time = time.time()
    
    # Calculate final metrics
    avg_tokens = tokens / len(test_examples)
    ratio = right / len(test_examples)
    latency = (end_time - start_time) / len(test_examples)
    
    print("#############################################MATH500 CD DRAFT#############################################")
    print("num of samples:", len(test_examples))
    print("avg tokens:", avg_tokens)
    print("avg ratio:", ratio)
    print("avg latency:", latency)
    
    # Save wrong answers to JSON
    with open("test_aime24.json", "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()