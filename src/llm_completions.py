import re
import numpy as np
from tqdm import tqdm
import gc
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM, 
    StoppingCriteria,
    StoppingCriteriaList,
    default_data_collator
)
from datasets import Dataset as hf_dataset



class CustomStoppingCriteria(StoppingCriteria):
    def __init__(self, stops = []):
        StoppingCriteria.__init__(self), 

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, stops = []):
        self.stops = stops
        for i in range(len(stops)):
            self.stops = self.stops[i]
            
               
class LLM_Generator:
       
    def __init__(
        self,
        model_name: str,
        messages_list: list,
        batch_size: int
    ):
        """Generate from Flan-T5 / Alpaca models.

        Args:
            messages_list: List of full contexts to generate from.
            model_name: Model type.
            temperature: Temperature to use.
            max_tokens: Maximum number of tokens to generate.
            top_p: P value for nucleus sampling.
            batch_size: Length of context to use.
        """
        self.batch_size = batch_size
        
        self.validate_model_name(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            padding_side="left"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name, 
            trust_remote_code=True,
            device_map="balanced_low_0"
        )
        self.device = f"cuda:{self.model.hf_device_map['lm_head']}"
        # self.max_length = self.tokenizer.model_max_length
        self.max_length = 256
        stop_word_list = ["}"]
        stop_words_ids = self.tokenizer(stop_word_list).input_ids
        self.stopping_criteria = StoppingCriteriaList(
            [CustomStoppingCriteria(stops = stop_words_ids)]
        )
        print(f"\nLoaded {model_name} for prompting...")
        print(f"\nModel max length: {self.max_length}")
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
            'flan_t5': 'google/flan-t5-xxl', 
            # 'mental_flan_t5': 't5-small',
            'mental_flan_t5': 'NEU-HAI/mental-flan-t5-xxl',
            'alpaca': 'declare-lab/flan-alpaca-gpt4-xl',
            'mental_alpaca': 'NEU-HAI/mental-alpaca',
        }
        assert model_type in model_dict, f"\n\nmodel_type not available... List of available models: {list(model_dict)}\n\n"
        self.model_name = model_dict[model_type]
        
        
    def tokenize_function(
        self,
        examples
    ):
        examples['prompt'] = [
            line for line in examples['prompt'] if len(line) > 0 and not line.isspace()
        ]
        return self.tokenizer(
            examples['prompt'],
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt"
        )
        
        
    def preprocess_data(self):
        self.tokenized_prompts = self.prompt_data.map(
            self.tokenize_function,
            batched=True,
            # batch_size=self.batch_size,
            remove_columns=list(self.prompt_data.features.keys()),
            load_from_cache_file=False,
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
        # top_k: float
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
                do_sample=True, 
                # top_k=top_k, 
                top_p=top_p,
                temperature =temperature,
                remove_invalid_values=True,
                early_stopping=True,
                return_dict_in_generate=True, 
                output_scores=True,
                stopping_criteria=self.stopping_criteria,
            )
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            ).detach().cpu()

            input_length = 1 if self.model.config.is_encoder_decoder else model_inputs['input_ids'].shape[1]
            generated_tokens = outputs.sequences[:, input_length:].detach().cpu()

            for i in range(len(generated_tokens)):
                token = []
                token_logprob = []
                for tok, score in zip(generated_tokens[i], transition_scores[i]):
                    token.append(self.tokenizer.decode(tok))
                    token_logprob.append(score.numpy().item())

                response = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)

                responses.append(response)
                tokens.append(token)
                token_logprobs.append(token_logprob)
            # gc.collect()
            # torch.cuda.empty_cache()
                
        return [responses, tokens, token_logprobs]
            

    




            
            
# class SyntheticDataGenerator:

#     def __init__(
#         self, 
#         model_type, 
#         prompt_df
#     ):
#         self.validate_model_name(model_type)
#         self.prompt_data = hf_dataset.from_pandas(prompt_df)
#         self.dst_data = prompt_df['dst'].tolist()
        
#         if self.model_name != "gpt3":
#             if self.model_name in [
#                 'google/flan-t5-xl', 
#                 'google/flan-t5-xxl', 
#                 'declare-lab/flan-alpaca-gpt4-xl', 
#                 'declare-lab/flan-alpaca-xxl'
#             ]:
#                 self.tokenizer = AutoTokenizer.from_pretrained(
#                     self.model_name, 
#                     padding_side="left"
#                 )
#                 self.model = AutoModelForSeq2SeqLM.from_pretrained(
#                     self.model_name, 
#                     trust_remote_code=True,
#                     device_map="balanced_low_0"
#                 )
        
#             elif self.model_name in [
#                 'mosaicml/mpt-7b-instruct', 
#                 'tiiuae/falcon-7b-instruct', 
#                 'facebook/opt-6.7b',
#                 'EleutherAI/gpt-neo-2.7B'
#             ]:
#                 self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
#                 self.model = AutoModelForCausalLM.from_pretrained(
#                     self.model_name, 
#                     trust_remote_code=True,
#                     torch_dtype=torch.float16,
#                     device_map="balanced_low_0"
#                 )
#                 if self.model_name == "EleutherAI/gpt-neo-2.7B":
#                     self.tokenizer.pad_token = self.tokenizer.eos_token
#                     self.model.resize_token_embeddings(len(self.tokenizer))
            
#             elif self.model_name == 'decapoda-research/llama-7b-hf':
#                 self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, padding_side="left")
#                 self.model = LlamaForCausalLM.from_pretrained(
#                     self.model_name, 
#                     trust_remote_code=True,
#                     device_map="balanced_low_0"
#                 )
#                 self.tokenizer.pad_token = "[PAD]"
#                 self.model.resize_token_embeddings(len(self.tokenizer))
                
#             if self.model_name in ['decapoda-research/llama-7b-hf', 'facebook/opt-6.7b']:
#                 self.max_length = self.model.config.max_position_embeddings
#             else:
#                 self.max_length = self.tokenizer.model_max_length
#             self.device = f"cuda:{self.model.hf_device_map['lm_head']}"


#             stop_word_list = ["[end_of_dialog]"]
#             stop_words_ids = self.tokenizer(stop_word_list).input_ids
#             self.stopping_criteria = StoppingCriteriaList(
#                 [CustomStoppingCriteria(stops = stop_words_ids)]
#             )
        
#             print(f"\nLoaded {self.model_name} for prompting...")
#             print(f"\nModel max length: {self.max_length}")
#             print(f"\nDevice: {self.device}")

#         self.preprocess_data()



    



#     def tokenize_function(
#         self,
#         examples
#     ):
#         examples['prompt'] = [
#             line for line in examples['prompt'] if len(line) > 0 and not line.isspace()
#         ]
#         return self.tokenizer(
#             examples['prompt'],
#             padding=True,
# #             padding='max_length',
# #             truncation=True,
# #             max_length=self.max_length,
#             add_special_tokens=False,
#             return_tensors="pt"
#         )



#     def preprocess_data(self):
#         if self.model_name != "gpt3":
#             self.tokenized_prompts = self.prompt_data.map(
#                 self.tokenize_function,
#                 batched=True,
#                 remove_columns=list(self.prompt_data.features.keys()),
#                 load_from_cache_file=True,
#             )
#             self.dataloader = DataLoader(
#                 self.tokenized_prompts, 
#                 shuffle=False, 
#                 collate_fn=default_data_collator, 
#                 batch_size=8
#             )
#         else:
#             self.tokenized_prompts = None
#         print(f"\nPreprocessed prompts...\ndataloader size: {len(self.dataloader)}\n")
            
        
        
#     def text_completion(self):
#         responses = []
#         tokens = []
#         token_logprobs = []
# #         top_logprobs = []
        
#         if self.model_name == "gpt3":
            
#             @retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(10))
#             def completion_with_backoff(**kwargs):
#                 return openai.Completion.create(**kwargs)

#             for prompt in tqdm(self.prompt_data, desc='Generating response...'):
#                 response = completion_with_backoff(
#                     model="text-davinci-003", 
#                     prompt=prompt, 
#                     max_tokens=200,
#                     top_p=0.4,
#                     logprobs=5,
#                     stop="[end_of_dialog]"
#                 )
#                 responses.append(response['choices'][0]['text'])
#                 tokens.append(response['choices'][0]['logprobs']['tokens'])
#                 token_logprobs.append(response['choices'][0]['logprobs']['token_logprobs'])
# #                 top_logprobs.append(response['choices'][0]['logprobs']['top_logprobs'])
    
#         else:
#             for i, model_inputs in enumerate(tqdm(self.dataloader, desc='Generating response...')):
#                 model_inputs = {key:torch.tensor(val).to(self.device) for key, val in model_inputs.items()}
#                 outputs = self.model.generate(
#                     **model_inputs, 
#                     max_new_tokens=150,
#                     do_sample=True, 
#                     top_k=10, 
#                     top_p=0.8,
#                     remove_invalid_values=True,
#                     early_stopping=True,
#                     return_dict_in_generate=True, 
#                     output_scores=True,
#                     stopping_criteria=self.stopping_criteria,
#                 )
#                 transition_scores = self.model.compute_transition_scores(
#                     outputs.sequences, outputs.scores, normalize_logits=True
#                 ).detach().cpu()

#                 input_length = 1 if self.model.config.is_encoder_decoder else model_inputs['input_ids'].shape[1]
#                 generated_tokens = outputs.sequences[:, input_length:].detach().cpu()

#                 for i in range(len(generated_tokens)):
#                     token = []
#                     token_logprob = []
#                     for tok, score in zip(generated_tokens[i], transition_scores[i]):
#                         token.append(self.tokenizer.decode(tok))
#                         token_logprob.append(score.numpy().item())

#                     response = self.tokenizer.decode(generated_tokens[i], skip_special_tokens=True)

#                     responses.append(response)
#                     tokens.append(token)
#                     token_logprobs.append(token_logprob)
#                 gc.collect()
#                 torch.cuda.empty_cache()
                    
#             return [responses, tokens, token_logprobs]