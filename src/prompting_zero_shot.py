import os
import random
import argparse
import pandas as pd
from utils import set_random_seed
from prompts import (
    CHAT_MODEL_ROLE,
    DEPRESSION_MARDS,
    DEPRESSION_PHQ9,
    DEPRESSION,
    DEPRESSION_LLAMA,
    ANXIETY_BAI,
    ANXIETY_HAMILTON,
    ANXIETY,
    ANXIETY_LLAMA,
    COMORBIDITY,
    COMORBIDITY_LLAMA,
)
import pprint
pp = pprint.PrettyPrinter(indent=4)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed', 
        type=int, 
        default=0, 
        help='Seed used to reproduce results.'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default="./data/test/full_test.csv",
        help="path to dataset file."
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default="gpt-3.5-turbo",
        choices=["gpt-3.5-turbo", "gpt-4", "mental_llama_chat_7b", "mental_llama_chat_13b", "llama_chat_7b", "llama_chat_13b", "llama_chat_70b"],
        help="type of model to use for prompting."
    )
    parser.add_argument(
        '--prompt_type', 
        type=str, 
        default="depression",
        choices=[
            'depression_mards', 'depression_phq9', 'depression', 'depression_llama', 
            'anxiety_bai', 'anxiety_hamilton', 'anxiety', 'anxiety_llama', 
            'comorbidity', 'comorbidity_llama'
        ],
        help='Type of prompt to use.'
    )
    parser.add_argument(
        '--version', 
        type=int, 
        default=1,
        help="Version of data"
    )
    parser.add_argument(
        '--result_dir', 
        type=str, 
        default="./results/zero_shot",
        help="path to dataset file."
    )
    return parser.parse_args()



if __name__ == "__main__":
    
    print(f"\nProcess ID: {os.getpid()}\n")
    
    # Read Args
    args = parse_config()

    # Set Seed
    set_random_seed(args.seed)

    print("\nArguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("\n")
    
    llm_prompt = {
        'depression_mards': DEPRESSION_MARDS, 
         'depression_phq9': DEPRESSION_PHQ9, 
         'depression': DEPRESSION, 
         'depression_llama': DEPRESSION_LLAMA,
         'anxiety_bai': ANXIETY_BAI, 
         'anxiety_hamilton': ANXIETY_HAMILTON, 
         'anxiety': ANXIETY,
         'anxiety_llama': ANXIETY_LLAMA,
         'comorbidity': COMORBIDITY,
         'comorbidity_llama': COMORBIDITY_LLAMA
    }[args.prompt_type]
    print(f"Using prompt {args.prompt_type}:\n{llm_prompt}\n\n")
    
    if args.prompt_type in ['depression_mards', 'depression_phq9', 'anxiety_bai', 'anxiety_hamilton']:
        max_tokens = 1024
    else:
        max_tokens = 48
    prompt_data = pd.read_csv(args.data_path)
    
    filename = f"zero_shot_{args.prompt_type}_{args.model}_seed_{args.seed}_v{args.version}"
    old_result_file = os.path.join(args.result_dir, f"{filename}_old.csv")
    if os.path.isfile(old_result_file):
        print("Found existing results")
        result_data = pd.read_csv(old_result_file)
        ids = result_data[result_data[f'results_{args.prompt_type}_{args.model}'].isnull()]['id'].tolist()
        prompt_data = prompt_data[prompt_data['id'].isin(ids)].reset_index(drop=True)
        result_file = os.path.join(args.result_dir, f"{filename}_new.csv")
    else:
        result_file = os.path.join(args.result_dir, f"{filename}.csv")
        
    # prompt_data = prompt_data[:100]
    print(f"\nsize of prompt data: {prompt_data.shape}")
    print(f"\nresult_file: {result_file}")
    
    # OPENAI Chat Models
    if args.model in ['gpt-3.5-turbo', 'gpt-4']:
        
        import asyncio
        from openai_completions import generate_from_openai_chat_completion

        input = []
        for text in prompt_data["text"].tolist():
            input.append(
                [
                    {"role": "system", "content": CHAT_MODEL_ROLE},
                    {"role": "user", "content": llm_prompt + text + "```"},
                ]
            )
        index = random.randint(0, len(input))
        pp.pprint(f"\nSample Input: {input[index]}")
                
        print(f"\n\nQuerying OpenAI {args.model}:\n")
        predictions = asyncio.run(
            generate_from_openai_chat_completion(
                messages_list=input,
                model=args.model,
                temperature=0,
                top_p=0.95,
                max_tokens=max_tokens,
                api_key="OPENAI_API_KEY",
                # org_key="OPENAI_ORG_KEY",
                requests_per_minute=30,
            )
        )
    
    else:
        from llm_completions import LLM_Generator

        input = []
        for text in prompt_data["text"].tolist():
            input.append({'prompt': f"Post: {text}\nQuestion: {llm_prompt}\nAnswer: "})
        
        index = random.randint(0, len(input))
        print(f"\nSample Input: {input[index]['prompt']}")
              
        generator = LLM_Generator(model_name=args.model, messages_list=input, batch_size=4)

        predictions = generator.text_completion(
            temperature=1,
            max_tokens=max_tokens,
            top_p=0.95,
        )
        
    # for prediction in predictions:
    #     print(f"{prediction}\n")
        
    prompt_data[f"results_{args.prompt_type}_{args.model}"] = predictions
    prompt_data.to_csv(result_file, index=False)