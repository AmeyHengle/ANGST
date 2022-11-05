import openai
import os
import json
import constants as const
from loguru import logger

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