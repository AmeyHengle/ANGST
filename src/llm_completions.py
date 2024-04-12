import re
import numpy as np
from tqdm import tqdm
import gc
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    default_data_collator
)
from datasets import Dataset as hf_dataset


  
class LLM_Generator:
       
    def __init__(
        self,
        model_name: str,
        chat_input: bool,
        messages_list: list,
        batch_size: int
    ):
        """Generate from Flan-T5 / Alpaca models.

        Args:
            messages_list: List of full contexts to generate from.
            model_name: Model type.
            chat_input: Whether to use chat input or not.
            temperature: Temperature to use.
            max_tokens: Maximum number of tokens to generate.
            top_p: P value for nucleus sampling.
            batch_size: Length of context to use.
        """
        self.batch_size = batch_size
        if chat_input:
            self.data_type = "chat"
            self.tokenize_function = self.tokenize_chat
        else:
            sel.data_type = "text"
            self.tokenize_function = self.tokenize_text

        self.validate_model_name(model_name)
        print(self.model_name)
        
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Configure the model
        if any(["CausalLM" in architecture for architecture in config.architectures]):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                device_map="auto",
            )
            
        elif any([("Seq2SeqLM" in architecture) or ("ConditionalGeneration" in architecture) for architecture in config.architectures]):
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                device_map="auto"
            )
                        
        else:
            raise ValueError(f'Model {self.model_name_or_path} is not adapted for the sequence generation task')

        print(f"\ncurrent device: {torch.cuda.current_device()}\n")
        device_mapping = self.model.hf_device_map
        print(device_mapping)
        if len(device_mapping.keys()) < 2:
            self.device = f"cuda:{list(device_mapping.values())[0]}"
        else:
            self.device = f"cuda:{device_mapping['lm_head']}"
            
        print(f"\nLoaded {model_name} for prompting...")
        print(f"\nDevice: {self.device}")
        
        self.prompt_data = hf_dataset.from_list(messages_list)
        print(f"\ndata: {self.prompt_data}")
        self.preprocess_data()
        print("\ndata processed.\n")
        
        
    def validate_model_name(
        self, 
        model_type
    ):
        model_dict = {
            'mental_llama_chat_7b': 'klyang/MentaLLaMA-chat-7B-hf', 
            'mental_llama_chat_13b': 'klyang/MentaLLaMA-chat-13B', 
            'llama_chat_7b': '/data/models/huggingface/meta-llama/Llama-2-7b-chat-hf', 
            'llama_chat_13b': '/data/models/huggingface/meta-llama/Llama-2-13b-chat-hf', 
            'llama_chat_70b': '/data/models/huggingface/meta-llama/Llama-2-70b-chat-hf'
        }
        assert model_type in model_dict, f"\n\nmodel_type not available... List of available models: {list(model_dict)}\n\n"
        self.model_name = model_dict[model_type]


        
    def tokenize_text(
        self,
        examples
    ):
        examples['prompt'] = [
            line for line in examples['prompt'] if len(line) > 0 and not line.isspace()
        ]
        tokenized_examples = self.tokenizer(
            examples['prompt'],
            padding=True,
            truncation=True,
        )
        return tokenized_examples
    
    
    def tokenize_chat(
        self,
        examples
    ):
        tokenized_examples = self.tokenizer.apply_chat_template(
            examples['prompt'], 
            tokenize=True, 
            add_generation_prompt=True, 
        )
        return tokenized_examples



    def preprocess_data(self):
        self.tokenized_prompts = self.prompt_data.map(
            self.tokenize_function,
            batched=True,
            remove_columns=list(self.prompt_data.features.keys()),
            load_from_cache_file=True,
            desc=f"Tokenizing {self.data_type} data"
        )
        print(f"\ntokenized_data: {self.tokenized_prompts}\n")
        self.dataloader = DataLoader(
            self.tokenized_prompts, 
            shuffle=False, 
            collate_fn=default_data_collator, 
            batch_size=self.batch_size
        )
        print(f"\nPreprocessed prompts...\ndataloader size: {len(self.dataloader)}\n")
        
        
        
    def text_completion(
        self,
        temperature: float,
        max_tokens: int,
        top_p: float,
    ):
        """
        Args:
            temperature: Temperature to use.
            max_tokens: Maximum number of tokens to generate.
            top_p: P value for nucleus sampling.
            top_k: K value for nucleus sampling. 
        Returns:
            List of generated responses.
        """
            
        responses = []
        tokens = []
        token_logprobs = []
        
        for i, model_inputs in enumerate(tqdm(self.dataloader, desc='Generating response...')):
            model_inputs = {key:torch.tensor(val).to(self.device) for key, val in model_inputs.items()}
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_tokens,
                num_beams=1,
                do_sample=True, 
                top_p=top_p,
                temperature =temperature,
                remove_invalid_values=True,
                early_stopping=False,
                return_dict_in_generate=True, 
                output_scores=False,
            )
            input_length = 1 if self.model.config.is_encoder_decoder else model_inputs['input_ids'].shape[1]
            sequences = [self.tokenizer.decode(sequence, skip_special_tokens=True) for sequence in outputs.sequences[:, input_length:].detach().cpu()]
            responses.extend(sequences)
            gc.collect()
            torch.cuda.empty_cache()
        return responses
    
    
    # tokenizer = AutoTokenizer.from_pretrained(
    #     model_name, 
    #     padding_side="left",
    # )