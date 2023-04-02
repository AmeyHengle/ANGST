import openai
import os
import json
import requests
import constants as const
from loguru import logger
from utils import set_random_seed

SEED = const.RANDOM_STATE
set_random_seed(SEED)

# --------------------------------------------------------------------------------------------------------------
# GPT3 Code
# --------------------------------------------------------------------------------------------------------------
api_key = None
with open(const.OPENAI_CREDS, 'r') as f:
    api_key = json.load(f)['api_key']
openai.api_key = api_key

def gpt(
    prompt,
    model=const.CONFIG_GPT['model'],
    temperature=const.CONFIG_GPT['temperature'],
    max_tokens=const.CONFIG_GPT['max_tokens'],
    top_p=const.CONFIG_GPT['top_p'],
    frequency_penalty=const.CONFIG_GPT['frequency_penalty'],
    presence_penalty=const.CONFIG_GPT['presence_penalty'],
    stop=const.CONFIG_GPT['stop']
):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop    
    )
    
    if response:
        try:
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.debug(f"Exception: {e}")
            return None
        

# --------------------------------------------------------------------------------------------------------------
# BLOOM Model inference using HuggingFace API
# --------------------------------------------------------------------------------------------------------------

def bloom(
    prompt,
    api_url=const.CONFIG_BLOOM['api_url'],
    temperature=const.CONFIG_BLOOM['temperature'],
    max_tokens=const.CONFIG_BLOOM['max_tokens'],
    top_p=const.CONFIG_BLOOM['top_p'],
    frequency_penalty=const.CONFIG_BLOOM['frequency_penalty'],
    presence_penalty=const.CONFIG_BLOOM['presence_penalty'],
    stop_sequence=const.CONFIG_BLOOM['stop_sequence']
):
    try:
        """
        Set your Hugging Face token as a environment variable.

        Eg:
        `export HF_TOKEN="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"`

        """
        HF_TOKEN = os.environ.get("HF_TOKEN") 
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        input_json = {
            "inputs": prompt,
            "parameters": {
                "top_p": top_p,
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "stop": stop_sequence,
                "return_full_text": False
            }, 
            "options": {
                "use_cache": True,
                "wait_for_model":True
                },
        }

        response = requests.post(api_url, headers=headers, json=input_json)
        output = response.json()
        generated_text = output[0]['generated_text']
        generated_text = generated_text.split("\nQ:")[0].strip()
        return generated_text

    except Exception as e:
        logger.debug(f"Exception: {e}")
        raise Exception(f"Exception: {e}")
        return None

# --------------------------------------------------------------------------------------------------------------
# Define Headers
# --------------------------------------------------------------------------------------------------------------
header1 = """Each item in the following list contains a reddit Post and the respective Mental Health Disorder.
The Disorder is one of ’Depression’, ’Anxiety’, ’Comorbid’ or ’Normal’.
"""


# --------------------------------------------------------------------------------------------------------------
# Define Prompt Templates
# --------------------------------------------------------------------------------------------------------------
prompt_template1 = lambda text, label: f"\nPost: {text}\nDisorder: {label}"

prompt_template2 = lambda text, label: f"\n{text}Disorder:{label}"