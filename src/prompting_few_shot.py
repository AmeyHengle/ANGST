import os
import argparse
import pandas as pd
import random
import asyncio
from utils import set_random_seed
from openai_completions import (
    generate_from_openai_completion,
    generate_from_openai_chat_completion
)
from prompts import CHAT_MODEL_ROLE
   

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
        choices=["gpt-3.5-turbo", "gpt-4", "flan-t5", "flan-alpaca", "llama2"],
        help="type of model to use for prompting."
    )
    parser.add_argument(
        '--prompt_type', 
        type=str, 
        default="depression",
        choices=['depression', 'anxiety', 'comorbidity'],
        help='Type of prompt to use.'
    )
    parser.add_argument(
        '--num_examples_per_label', 
        type=int, 
        default=2,
        help='Number of examples per label for in-context learning'
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
    print(f"Using data for {args.prompt_type}\n\n")
    
    max_tokens = 64
    prompt_data = pd.read_csv(args.data_path)
        
    old_result_file = os.path.join(args.result_dir, f"few_shot_{args.prompt_type}_{args.model}_num_examples_ss_{args.num_examples_per_label}_seed_{args.seed}_old.csv")
    if os.path.isfile(old_result_file):
        print("Found existing results")
        result_data = pd.read_csv(old_result_file)
        ids = result_data[result_data[f'results_{args.prompt_type}_{args.model}'].isnull()]['id'].tolist()
        prompt_data = prompt_data[prompt_data['id'].isin(ids)].reset_index(drop=True)
        result_file = os.path.join(args.result_dir, f"few_shot_{args.prompt_type}_{args.model}_num_examples_ss_{args.num_examples_per_label}_seed_{args.seed}_new.csv")
    else:
        result_file = os.path.join(args.result_dir, f"few_shot_{args.prompt_type}_{args.model}_num_examples_ss_{args.num_examples_per_label}_seed_{args.seed}.csv")
    print(f"\nsize of prompt data: {prompt_data.shape}")
    
    if args.model in ['gpt-3.5-turbo', 'gpt-4', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002']:
        
        # OPENAI Chat Models
        if args.model in ['gpt-3.5-turbo', 'gpt-4']:
            prompting_function = generate_from_openai_chat_completion
            input = []
            for text in prompt_data[f'few_shot_prompt_{args.prompt_type}'].tolist():
                input.append(
                    [
                        {"role": "system", "content": CHAT_MODEL_ROLE},
                        {"role": "user", "content": text},
                    ]
                )

        # OPENAI Completion Models
        else:
            prompting_function = generate_from_openai_completion
            input = []
            for text in prompt_data[f'few_shot_prompt_{args.prompt_type}'].tolist():
                if "mental_llm" in args.prompt_type:
                    input.append(text + llm_prompt)
                else:
                    input.append(llm_prompt + text + "```")

        index = random.randint(0, len(input))
        print(f"\nSample Input: {input[index]}")
        
        print("\n\nQuerying OpenAI:\n")
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

    prompt_data[f"results_{args.prompt_type}_{args.model}"] = predictions
    prompt_data.to_csv(result_file, index=False)