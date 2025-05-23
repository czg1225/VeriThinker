import json
import os
import re


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
    
def get_examples(split):
    path = os.path.join("data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE_gt = re.compile(r"#### (\-?[0-9\.\,]+)")
ANS_RE_qwq = re.compile(r"boxed\{(.*?)\}")

INVALID_ANS = "[invalid]"

def extract_answer_gt(completion):
    match = ANS_RE_gt.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

def extract_answer_qwq(completion):
    match = ANS_RE_qwq.search(completion)
    if match:
        match_str = match.group(1).strip()
        # 先移除所有可能的 % 和 逗号
        match_str = match_str.replace(",", "").replace("%", "").replace("\\", "").replace("$", "")
        return match_str
    else:
        return INVALID_ANS

def extract_answer_llm(text):
    """
    从文本中提取最后一个包含数字的字符串，并只保留数字、小数点和正负号
    
    Args:
        text (str): 输入文本
        
    Returns:
        str: 处理后的数字字符串，如果没找到则返回 "INVALID_ANS"
    """
    # 匹配包含数字的字符串
    number_strings = re.findall(r'\S*\d+\S*', text)
    
    if not number_strings:
        return "INVALID_ANS"
    
    # 获取最后一个匹配的字符串
    last_number_string = number_strings[-1]
    
    # 只保留数字、小数点和正负号
    cleaned_number = ''.join(char for char in last_number_string 
                           if char.isdigit() or char in '.-')
    
    # 处理可能出现的多个小数点或正负号
    # 只保留第一个小数点
    if cleaned_number.count('.') > 1:
        first_dot_index = cleaned_number.index('.')
        cleaned_number = cleaned_number[:first_dot_index + 1] + \
                        cleaned_number[first_dot_index + 1:].replace('.', '')
    
    # 只保留最前面的正负号
    if cleaned_number.startswith('-'):
        cleaned_number = '-' + cleaned_number[1:].replace('-', '')
    else:
        cleaned_number = cleaned_number.replace('-', '')
    
    return cleaned_number

def extract_all_boxed_content(text):
    results = []
    start = 0

    while True:
        # Find the next occurrence of \boxed{
        start = text.find(r"\boxed{", start)
        if start == -1:
            break  # No more \boxed{ found

        brace_count = 0
        result = []
        i = start

        while i < len(text):
            char = text[i]
            result.append(char)

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1

            # Stop when the braces are balanced
            if brace_count == 0 and result[-1] == '}':
                break

            i += 1

        # Append the matched content
        results.append(''.join(result))
        start = i + 1  # Move past the current match to find the next

    return results

def is_correct(model_completion, gt_example):
    gt_answer = extract_answer_gt(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer_gt(model_completion) == gt_answer


