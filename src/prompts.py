CHAT_MODEL_ROLE = '''
You are a psychology professor working in the domain of mental health. 
You are well versed with The Diagnostic and Statistical Manual of Mental Disorders (DSM-5) Manual, which is the authoritative guide to the diagnosis of mental disorders.
'''

# ----------------------------------------- DEPRESSION PROMPTS -----------------------------------------

DEPRESSION_MARDS = '''
Below are the 10 symptoms of depression and their definitions according to the Montgomery and Åsberg Depression Rating Scale.
As a psychologist, read the social media post and evaluate the severity score of each symptom on a scale of 0 to 6. 
The social media post is delimited with triple backticks.
Format your response as a JSON object with serial number of the 10 symptoms as keys.
Do not return anything else apart from the json object.

Symptoms:
1.Apparent Sadness :
Representing despondency, gloom and despair, (more than just ordinary transient low spirits) reflected in speech, facial expression, and posture
2.Reported sadness:
Representing reports of depressed mood, regardless of whether it is reflected in appearance or not. Includes low spirits, despondency or the feeling of being beyond help and without hope.
3.Inner tension:
Representing feelings of ill-defined discomfort, edginess, inner turmoil, mental tension mounting to either panic, dread or anguish.
4.Reduced sleep:
Representing the experience of reduced duration or depth of sleep compared to the subject's own normal pattern when well.
5.Reduced appetite:
Representing the feeling of a loss of appetite compared with when well.
6.Concentration Difficulties:
Representing difficulties in collecting one's thoughts mounting to incapacitating lack of concentration. Rate according to intensity, frequency, and degree of incapacity produced.
7.Lassitude Representing:
a difficulty getting started or slowness initiating and performing everyday activities
8.Inability to feel:
Representing the subjective experience of reduced interest in the surroundings, or activities that normally give pleasure. The ability to react with adequate emotion to circumstances or people is reduced.
9.Pessimistic thoughts:
Representing thoughts of guilt, inferiority, self-reproach, sinfulness, remorse and ruin
10.Suicidal thoughts:
Representing the feeling that life is not worth living, that a natural death would be welcome, suicidal thoughts, and preparations for suicide.

Post:
```
'''


DEPRESSION_PHQ9 = '''
Below are the 9 symptoms of depression according to the the Patient Health Questionnaire (PHQ-9)
As a psychologist, read the social media post and determine which of the 9 symptoms observed in the post.
The social media post is delimited with triple backticks.
Format your response as a JSON object with serial number of the 9 symptoms as keys and "yes" or "no" as values.
Do not return anything else apart from the json object.

Symptoms:
1.Little interest or pleasure in doing things
2.Feeling down, depressed, or hopeless.
3.Trouble falling or staying asleep, or sleeping too much.
4.Feeling tired or having little energy.
5.Poor appetite or overeating.
6.Feeling bad about yourself – or that you are a failure or have let yourself or your family down
7.Trouble concentrating on things, such as reading the newspaper or watching television.
8.Moving or speaking so slowly that other people could have noticed. Or the opposite – being so fidgety or restless that you have been moving around a lot more than usual.
9.Thoughts that you would be better off dead, or of hurting yourself in some way.

Post:
```
'''

DEPRESSION = '''
Assess whether the content of the post delimited by triple backticks aligns with the criteria for clinical depression as defined in DSM-5:
Format your response as a JSON object {'depression':''} with values strictly either 'yes' or 'no'.
Do not return anything else apart from the json object.

Post:
```
'''

# Add Text before prompt
DEPRESSION_MENTAL_LLM = '''
This person wrote this paragraph on social media. As a psychologist, read the post on social media
and answer the question.
Determine if the poster of this post is depressed.
'''

DEPRESSION_ZEROSHOT = '''
Given the below post, assess whether its content aligns with the criteria for clinical depression as defined in the DSM-5.
Format your response as a JSON object {'depression':''} with values either 'yes' or 'no'.

Post: {post}
Assesement:
'''

DEPRESSION_FEWSHOT = '''
Below are posts and their respective assessments based on the criteria for clinical depression as defined in the DSM-5.
Format your response as a JSON object {'depression':''} with values either 'yes' or 'no'.

Post: "Every night I lay in bed with intense fear and dread, and these feelings have persisted for a year."
Assessment: {'depression': 'yes'}

Post: "I have a cat named Whiskers and he loves playing with yarn."
Assessment: {'depression': 'no'}

Based on the above, assess the content of the following post:
Post: "sample post"
Assessment:
'''

DEPRESSION_FEWSHOT_LANGCHAIN = {
"few_shot_prefix": '''
Below are posts and their respective assessments based on the criteria for clinical depression as defined in the DSM-5.
Format your response as a JSON object {'depression': ''} with values either 'yes' or 'no'.
'''
,
"prompt_template": lambda post, label: f'''
Post: {post}
Assesement: {label}
'''
,
"few_shot_suffix": lambda post: f'''
Based on the above, assess the content of the following post:
Post: {post}
Assessment:
'''
}

# ----------------------------------------- ANXIETY PROMPTS -----------------------------------------

ANXIETY_BAI = '''
Below are  21 common symptoms of anxiety according to the Beck Anxiety Inventory (BAI).
As a psychologist, read the social media post and evaluate the severity score of each symptom on a scale of 0 to 4. 
The social media post is delimited with triple backticks.
Format your response as a JSON object with serial number of the 21 symptoms as keys.

1. Numbness or tingling
2. Feeling hot
3. Wobbliness in legs
4. Unable to relax
5. Fear of the worst happening
6. Dizzy or lightheaded
7. Heart pounding/racing
8. Unsteady
9. Terrified or afraid
10. Nervous
11. Feeling of choking
12. Hands trembling
13. Shaky/unsteady
14. Fear of losing control
15. Difficulty in breathing
16. Fear of dying
17. Feeling scared
18. Indigestion
19. Faint/lightheaded
20. Face flushed
21. Hot/cold sweats

Post:
```
'''



ANXIETY_HAMILTON = '''
Below are the 14 symptoms of anxiety and their definitions according to the Hamilton Anxiety Rating Scale.
As a psychologist, read the social media post and evaluate the severity score of each symptom on a scale of 0 to 4. 
The social media post is delimited with triple backticks.
Format your response as a JSON object with serial number of the 14 symptoms as keys.

Do not return anything else apart from the json object.
1 Anxious mood:
Worries, anticipation of the worst, fearful anticipation, irritability.
2 Tension:
Feelings of tension, fatigability, startle response, moved to tears
easily, trembling, feelings of restlessness, inability to relax.
3 Fears:
Of dark, of strangers, of being left alone, of animals, of traffic, of
crowds.
4 Insomnia:
Difficulty in falling asleep, broken sleep, unsatisfying sleep and fatigue
on waking, dreams, nightmares, night terrors.
5 Intellectual:
Difficulty in concentration, poor memory.
6 Depressed mood:
Loss of interest, lack of pleasure in hobbies, depression, early waking,
diurnal swing.
7 Somatic (muscular):
Pains and aches, twitching, stiffness, myoclonic jerks, grinding of
teeth, unsteady voice, increased muscular tone.
8 Somatic (sensory):
Tinnitus, blurring of vision, hot and cold flushes, feelings of weakness,
pricking sensation.
9 Cardiovascular symptoms:
Tachycardia, palpitations, pain in chest, throbbing of vessels, fainting
feelings, missing beat.
10 Respiratory symptoms:
Pressure or constriction in chest, choking feelings, sighing, dyspnea.
11 Gastrointestinal symptoms:
Difficulty in swallowing, wind abdominal pain, burning sensations,
abdominal fullness, nausea, vomiting, borborygmi, looseness of
bowels, loss of weight, constipation.
12 Genitourinary symptoms:
Frequency of micturition, urgency of micturition, amenorrhea,
menorrhagia, development of frigidity, premature ejaculation, loss of
libido, impotence.
13 Autonomic symptoms:
Dry mouth, flushing, pallor, tendency to sweat, giddiness, tension
headache, raising of hair.
14 Behavior at interview:
Fidgeting, restlessness or pacing, tremor of hands, furrowed brow,
strained face, sighing or rapid respiration, facial pallor, swallowing,
etc.

Post:
```
'''



ANXIETY = '''
Assess whether the content of the post delimited by triple backticks aligns with the criteria for clinical anxiety as defined in DSM-5:
Format your response as a JSON object {'anxiety':''} with values strictly either 'yes' or 'no'.
Do not return anything else apart from the json object.

Post:
```
'''



ANXIETY_MENTAL_LLM = '''
This person wrote this paragraph on social media. As a psychologist, read the post on social media
and answer the question.
Determine if the poster of this post has anxiety.
'''

ANXIETY_ZEROSHOT = '''
Given the below post, assess whether its content aligns with the criteria for clinical anxiety as defined in the DSM-5.
Format your response as a JSON object {'anxiety':''} with values either 'yes' or 'no'.

Post: {post}
Assesement: 
'''

ANXIETY_FEWSHOT = '''
Below are posts and their respective assessments based on the criteria for clinical anxiety as defined in the DSM-5.
Format your response as a JSON object {'anxiety':''} with values either 'yes' or 'no'.

Post: "Every night I lay in bed with intense fear and dread, and these feelings have persisted for a year."
Assessment: {'anxiety': 'yes'}

Post: "I have a cat named Whiskers and he loves playing with yarn."
Assessment: {'anxiety': 'no'}

Based on the above, assess the content of the following post:
Post: "sample post"
Assessment:
'''

ANXIETY_FEWSHOT_LANGCHAIN = {
"few_shot_prefix": '''
Below are posts and their respective assessments based on the criteria for clinical anxiety as defined in the DSM-5.
Format your response as a JSON object {'anxiety':''} with values either 'yes' or 'no'.
'''
,
"prompt_template": lambda post, label: f'''
Post: {post}
Assesement: {label}
'''
,
"few_shot_suffix": lambda post: f'''
Based on the above, assess the content of the following post:
Post: {post}
Assessment:
'''
}
# ----------------------------------------- DEPRESSION-ANXIETY COMORBIDITY PROMPTS -----------------------------------------

COMORBIDITY = """
This person wrote this paragraph on social media. As a psychologist, read the post on social media
and answer the following questions.
1. Does the poster of this post have anxiety?
2. Does if the poster of this post have depression?
Format your response as a JSON object {'depression':'' 'anxiety': } with values strictly either 'yes' or 'no'.
Do not return anything else apart from the json object.
"""


COMORBIDITY_FEWSHOT_1 = """
Below are posts and their respective assessments based on the criteria for clinical depression and clinical anxiety respectively as defined in the DSM-5.
Format your response as a JSON object {'depression': '', 'anxiety': ''} with values either 'yes' or 'no'.

Post: "Every night I lay in bed with intense fear and dread, and these feelings have persisted for a year."
Assessment: {'depression': 'yes', 'anxiety': 'yes'}

Post: "I have a cat named Whiskers and he loves playing with yarn."
Assessment: {'depression': 'no', 'anxiety': 'no'}

Based on the above, assess the content of the following post:
Post: "sample post"
Assessment:
"""


COMORBIDITY_FEWSHOT_2 = '''
Below are posts and their respective assessments based on the criteria for either of clinical depression, clinical anxiety, comorbidity (depression and anxiety) or none as defined in the DSM-5.
Format your response as a JSON object {'depression': '', 'anxiety': ''} with values either 'yes' or 'no'.

Post: "Every night I lay in bed with intense fear and dread, and these feelings have persisted for a year."
Assessment: {'label': 'depression'}

Post: "I have a cat named Whiskers and he loves playing with yarn."
Assessment: {'label': 'none'}

Based on the above, assess the content of the following post:
Post: "sample post"
Assessment:
'''

COMORBIDITY_FEWSHOT_LANGCHAIN = {
"few_shot_prefix": '''
Below are posts and their respective assessments based on the criteria for clinical depression and clinical anxiety respectively as defined in the DSM-5.
Format your response as a JSON object {'depression': '', 'anxiety': ''} with values either 'yes' or 'no'.
'''
,
"prompt_template": lambda post, label: f'''
Post: {post}
Assesement: {label}
'''
,
"few_shot_suffix": lambda post: f'''
Based on the above, assess the content of the following post:
Post: {post}
Assessment:
'''
}