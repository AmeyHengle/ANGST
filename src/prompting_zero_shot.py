import os
import argparse
import pandas as pd
import asyncio
from utils import set_random_seed
from openai_completions import (
    generate_from_openai_completion,
    generate_from_openai_chat_completion
)
from llm_completions import LLM_Generator
from prompts import (
    CHAT_MODEL_ROLE,
    DEPRESSION_MARDS,
    DEPRESSION_PHQ9,
    DEPRESSION,
    DEPRESSION_MENTAL_LLM,
    ANXIETY_BAI,
    ANXIETY_HAMILTON,
    ANXIETY,
    ANXIETY_MENTAL_LLM,
    COMORBIDITY,
)

# API_KEY = "OPENAI_API_KEY"
API_KEY = "SHRUTI_OPENAI_API_KEY"
# API_KEY = "ANDY_OPENAI_API_KEY"
# API_KEY = "JOEL_OPENAI_API_KEY"
print(f"\nUsing {API_KEY}\n")


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
        choices=["gpt-3.5-turbo", "gpt-4", "flan_t5", "mental_flan_t5", "alpaca", "mental_alpaca"],
        help="type of model to use for prompting."
    )
    parser.add_argument(
        '--prompt_type', 
        type=str, 
        default="depression",
        choices=['depression_mards', 'depression_phq9', 'depression', 'depression_mental_llm', 'anxiety_bai', 'anxiety_hamilton', 'anxiety', 'anxiety_mental_llm', 'comorbidity'],
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
         'depression_mental_llm': DEPRESSION_MENTAL_LLM, 
         'anxiety_bai': ANXIETY_BAI, 
         'anxiety_hamilton': ANXIETY_HAMILTON, 
         'anxiety': ANXIETY,
         'anxiety_mental_llm': ANXIETY_MENTAL_LLM,
         'comorbidity': COMORBIDITY,
    }[args.prompt_type]
    print(f"Using prompt {args.prompt_type}:\n{llm_prompt}\n\n")
    
    if args.prompt_type in ['depression_mards', 'depression_phq9', 'anxiety_bai', 'anxiety_hamilton']:
        max_tokens = 1024
    else:
        max_tokens = 64
    prompt_data = pd.read_csv(args.data_path)
    
    old_result_file = os.path.join(args.result_dir, f"zero_shot_{args.prompt_type}_{args.model}_seed_{args.seed}_v{args.version}_old.csv")
    if os.path.isfile(old_result_file):
        print("Found existing results")
        result_data = pd.read_csv(old_result_file)
        ids = result_data[result_data[f'results_{args.prompt_type}_{args.model}'].isnull()]['id'].tolist()
        prompt_data = prompt_data[prompt_data['id'].isin(ids)].reset_index(drop=True)
        result_file = os.path.join(args.result_dir, f"zero_shot_{args.prompt_type}_{args.model}_seed_{args.seed}_v{args.version}_new.csv")
    else:
        result_file = os.path.join(args.result_dir, f"zero_shot_{args.prompt_type}_{args.model}_seed_{args.seed}_v{args.version}.csv")
    print(f"\nsize of prompt data: {prompt_data.shape}")
    print(f"\nresult_file: {result_file}")
    
    if args.model in ['gpt-3.5-turbo', 'gpt-4', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002']:
        
        # OPENAI Chat Models
        if args.model in ['gpt-3.5-turbo', 'gpt-4']:
            prompting_function = generate_from_openai_chat_completion
            input = []
            for text in prompt_data["text"].tolist():
                if "mental_llm" in args.prompt_type:
                        input.append(
                        [
                            {"role": "system", "content": CHAT_MODEL_ROLE},
                            {"role": "user", "content": text + llm_prompt},
                        ]
                    )
                else:
                    input.append(
                        [
                            {"role": "system", "content": CHAT_MODEL_ROLE},
                            {"role": "user", "content": llm_prompt + text + "```"},
                        ]
                    )

        # OPENAI Completion Models
        else:
            prompting_function = generate_from_openai_completion
            input = []
            for text in prompt_data["text"].tolist():
                input.append(llm_prompt + text + "```")
        print(f"\nSample Input: {input[0]}")
        
        print(f"\n\nQuerying OpenAI {args.model}:\n")
        predictions = asyncio.run(
            prompting_function(
                messages_list=input,
                model=args.model,
                temperature=0,
                max_tokens=max_tokens,
                api_key=API_KEY,
                requests_per_minute=30,
            )
        )
    
    elif args.model in ['flan_t5', 'mental_flan_t5', 'alpaca', 'mental_alpaca']:

        input = []
        for text in prompt_data["text"].tolist():
            input.append({'prompt': llm_prompt + text + "```"})
        print(f"\nSample Input: {input[0]}")

        generator = LLM_Generator(model_name=args.model, messages_list=input, batch_size=2)

        predictions, _, _ = generator.text_completion(
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
        )
        
    else:
        print(f"\n\n Model {args.model} not supported...")

    prompt_data[f"results_{args.prompt_type}_{args.model}"] = predictions
    prompt_data.to_csv(result_file, index=False)