import os
import argparse
import pandas as pd
import random
from utils import set_random_seed
   
   
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
        choices=["gpt-3.5-turbo", "gpt-4", "mental_llama_chat_7b", "mental_llama_chat_13b"],
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
    print(f"Using data for {args.prompt_type}\n\n")
    
    max_tokens = 24
    prompt_data = pd.read_csv(args.data_path)
        
    old_result_file = os.path.join(args.result_dir, f"few_shot_{args.prompt_type}_{args.model}_num_examples_ss_{args.num_examples_per_label}_v{args.version}_seed_{args.seed}_older.csv")
    if os.path.isfile(old_result_file):
        print("Found existing results")
        result_data = pd.read_csv(old_result_file)
        ids = result_data[result_data[f'results_{args.prompt_type}_{args.model}'].isnull()]['id'].tolist()
        prompt_data = prompt_data[prompt_data['id'].isin(ids)].reset_index(drop=True)
        result_file = os.path.join(args.result_dir, f"few_shot_{args.prompt_type}_{args.model}_num_examples_ss_{args.num_examples_per_label}_v{args.version}_seed_{args.seed}_new.csv")
    else:
        result_file = os.path.join(args.result_dir, f"few_shot_{args.prompt_type}_{args.model}_num_examples_ss_{args.num_examples_per_label}_v{args.version}_seed_{args.seed}.csv")
    # prompt_data = prompt_data[:20]
    print(f"\nsize of prompt data: {prompt_data.shape}")
    print(f"\nresult_file: {result_file}")
    
    if args.model in ['gpt-3.5-turbo', 'gpt-4']:
        
        import asyncio
        from openai_completions import generate_from_openai_chat_completion
        from prompts import CHAT_MODEL_ROLE

        input = []
        for text in prompt_data['prompt'].tolist():
        # for text in prompt_data[f'few_shot_prompt_{args.prompt_type}'].tolist():
            input.append(
                [
                    {"role": "system", "content": CHAT_MODEL_ROLE},
                    {"role": "user", "content": text},
                ]
            )
        index = random.randint(0, len(input))
        print(f"\nSample Input: {input[index]}")
        
        print("\n\nQuerying OpenAI:\n")
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
        
    elif args.model in ['mental_llama_chat_7b', 'mental_llama_chat_13b']:
        from llm_completions import LLM_Generator

        input = [{'prompt': text} for text in prompt_data[f'few_shot_prompt_{args.prompt_type}'].tolist()]
        index = random.randint(0, len(input))
        print(f"\nSample Input: {input[index]['prompt']}")

        generator = LLM_Generator(model_name=args.model, messages_list=input, batch_size=1)

        predictions = generator.text_completion(
            temperature=1,
            max_tokens=max_tokens,
            top_p=0.95,
        )
        
    else:
        print(f"\n\n Model {args.model} not supported...")

    prompt_data[f"results_{args.prompt_type}_{args.model}"] = predictions
    prompt_data.to_csv(result_file, index=False)