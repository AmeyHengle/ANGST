### Overview:
The .csv files in this directory contain few shot prompts for each datapoint in the test set (2872 in total).\
These files can be directly used for GPT inference. 

#### depression.csv
column: few_shot_prompt_depression \
topk: 4 (2 from depression and normal categories each) \
data snapshot:
```
Below are posts and their respective assessments based on the criteria for clinical depression as defined in the DSM-5.
Format your response as a JSON object {'depression': ''} with values either 'yes' or 'no'.

Post: i fucking hate myself. i want to become a surgeon but at this rate, thats no where near possible. i just scored a 53 on my biology final and i was so close to a b and was actually kind of confident. now im going to have a c on my semester final. i studied hard for this test but im just a fucking idiot. someone also crashed their car into mine and took advantage of my age with the whole insurance process so thats fucking great.
Assesement: {'depression': 'no'}

Post: hi guys! i finally feel like im doing a little bit better, especially since i met someone special a couple of months ago. so now, here i am leaving not only this subreddit, but also all of the self harm subreddits im in because i cant really relate anymore, and some things are a bit triggering. good luck to everyone who is still struggling. its really fucken tough but that makes you the strongest kind of people for being able to endure all of this pain, even if you want to give up sometimes. i love all of you, and every single one of you deserves living just as much as anyone else.
Assesement: {'depression': 'no'}

Post: so, ive been inactive on this account for a while, but im here to just say stuff again like i did in my other two posts or whatever. it seems like ive just started doing it because i got bored. it gives me something to focus on instead of having to think about everything happening at school and home. its also a stress relief for me. whenever im pissed or stressed ill make a scratch or two and it helps a lot. it doesnt seem like theres anything else that works as well as that. i did a dumb thing yet i dont regret it, but i still dont know why i did it. so, i cant remember the saying but its some retarded self harm thing basically meaning you only do it up the thinner part of your arm if you get what i mean if youre looking for attention. i think this made me start to do it more along my arm and a lot deeper. i cant post pictures, but i made about 6 cuts maybe 2 inches long each with a razor and it bled a lot. its scabbed up now, so thats god. anyway, the reason i did it was because my character in a roleplay game got permanently killed. yeah, thats why ive decided to scar myself. i dont know why i did that but i dont feel like i regret it at all. i feel relieved but i dont know why. is this normal? why cant i stop doing this? only two people know about my scars one of them i hate, one of them i never meet so its difficult for me to do or say anything.
Assesement: {'depression': 'yes'}

Post: i self harmed tonight more times than i can count. i work in the medical field and im always trying to come up with solutions on how to end it all. all the pain and heartache. im 30 years old and nothing is getting better. i self harm because i feel like i deserve it. the scars on the outside some how feel as if they belong there because of how fucked up i feel on the inside. it isnt fair. i dont want to be here.
Assesement: {'depression': 'yes'}

Based on the above, assess the content of the following post:
Post: I cut myself for the first time in a year today. and hated that I still loved it. The burning sensation from the blade on my arm and leg. I didnt even cut deep, I barely bled, but the feeling was still almost the same, but without the kick of seeing my own blood trail down my limbs staining the covers in my bed. I told myself that I was only going to cut a few times, but here I am with surely thirty new scars on my body. I fucking hate myself.
Assessment:
```

#### anxiety.csv
column: few_shot_prompt_anxiety \
topk: 4 (2 each from anxiety and normal categories respectively) \
data snapshot: 
```
Below are posts and their respective assessments based on the criteria for clinical anxiety as defined in the DSM-5.
Format your response as a JSON object {'anxiety':''} with values either 'yes' or 'no'.

Post: i fucking hate myself. i want to become a surgeon but at this rate, thats no where near possible. i just scored a 53 on my biology final and i was so close to a b and was actually kind of confident. now im going to have a c on my semester final. i studied hard for this test but im just a fucking idiot. someone also crashed their car into mine and took advantage of my age with the whole insurance process so thats fucking great.
Assesement: {'anxiety': 'no'}

Post: please. please. please. i thought about self harming the other day, i was so anxious i had to take my benzodiazepines but ended up taking double dose, which has got me drowsy and groggy throughout the day. my boyfriend took care of me today and im so fucking grateful for him, i love that man, hes my everything, my rock. i dont want to spiral, ive been doing so well for 6 months. so. fucking. well.
Assesement: {'anxiety': 'yes'}

Post: ive been so starved for connection with people that today when i went to dinner with a bunch of new people for my friends birthday, i felt like i did everything wrong. i spoke too much, didnt listen enough, wore too revealing of a dress. i can tell that some of the thoughts are twisted by my brain but i still really hate myself and probably wont stop beating myself up for a long time. i felt like i was annoying and they all think im the worst. but im sure they dont even care enough to give me that much thought
Assesement: {'anxiety': 'yes'}

Post: hi guys! i finally feel like im doing a little bit better, especially since i met someone special a couple of months ago. so now, here i am leaving not only this subreddit, but also all of the self harm subreddits im in because i cant really relate anymore, and some things are a bit triggering. good luck to everyone who is still struggling. its really fucken tough but that makes you the strongest kind of people for being able to endure all of this pain, even if you want to give up sometimes. i love all of you, and every single one of you deserves living just as much as anyone else.
Assesement: {'anxiety': 'no'}

Based on the above, assess the content of the following post:
Post: I cut myself for the first time in a year today. and hated that I still loved it. The burning sensation from the blade on my arm and leg. I didnt even cut deep, I barely bled, but the feeling was still almost the same, but without the kick of seeing my own blood trail down my limbs staining the covers in my bed. I told myself that I was only going to cut a few times, but here I am with surely thirty new scars on my body. I fucking hate myself.
Assessment:
```

#### comorbidity.csv
column: few_shot_prompt_comorbidity \
topk: 4 (1 each from depression, anxiety, comorbidity, and normal categories respectively) \
data snapshot: 
```
Below are posts and their respective assessments based on the criteria for clinical depression and clinical anxiety respectively as defined in the DSM-5.
Format your response as a JSON object {'depression': '', 'anxiety': ''} with values either 'yes' or 'no'.

Post: ive been so starved for connection with people that today when i went to dinner with a bunch of new people for my friends birthday, i felt like i did everything wrong. i spoke too much, didnt listen enough, wore too revealing of a dress. i can tell that some of the thoughts are twisted by my brain but i still really hate myself and probably wont stop beating myself up for a long time. i felt like i was annoying and they all think im the worst. but im sure they dont even care enough to give me that much thought
Assesement: {'depression': 'no', 'anxiety': 'yes'}

Post: so, ive been inactive on this account for a while, but im here to just say stuff again like i did in my other two posts or whatever. it seems like ive just started doing it because i got bored. it gives me something to focus on instead of having to think about everything happening at school and home. its also a stress relief for me. whenever im pissed or stressed ill make a scratch or two and it helps a lot. it doesnt seem like theres anything else that works as well as that. i did a dumb thing yet i dont regret it, but i still dont know why i did it. so, i cant remember the saying but its some retarded self harm thing basically meaning you only do it up the thinner part of your arm if you get what i mean if youre looking for attention. i think this made me start to do it more along my arm and a lot deeper. i cant post pictures, but i made about 6 cuts maybe 2 inches long each with a razor and it bled a lot. its scabbed up now, so thats god. anyway, the reason i did it was because my character in a roleplay game got permanently killed. yeah, thats why ive decided to scar myself. i dont know why i did that but i dont feel like i regret it at all. i feel relieved but i dont know why. is this normal? why cant i stop doing this? only two people know about my scars one of them i hate, one of them i never meet so its difficult for me to do or say anything.
Assesement: {'depression': 'yes', 'anxiety': 'no'}

Post: i fucking hate myself. i want to become a surgeon but at this rate, thats no where near possible. i just scored a 53 on my biology final and i was so close to a b and was actually kind of confident. now im going to have a c on my semester final. i studied hard for this test but im just a fucking idiot. someone also crashed their car into mine and took advantage of my age with the whole insurance process so thats fucking great.
Assesement: {'depression': 'no', 'anxiety': 'no'}

Post: after years of struggling, i finally had the courage to seek professional help. i was diagnosed with major depression, severe anxiety and ptsd. the doctor saw my cuts and as she held my arm, i almost sobbed in front of her. she asked me what made me decide to seek help. i was quiet for a moment and it came to me that i wanna get better. i want to get rid of the voices that tell me i dont belong in this world. i wish i will get better. lets see.
Assesement: {'depression': 'yes', 'anxiety': 'yes'}

Based on the above, assess the content of the following post:
Post: I cut myself for the first time in a year today. and hated that I still loved it. The burning sensation from the blade on my arm and leg. I didnt even cut deep, I barely bled, but the feeling was still almost the same, but without the kick of seeing my own blood trail down my limbs staining the covers in my bed. I told myself that I was only going to cut a few times, but here I am with surely thirty new scars on my body. I fucking hate myself.
Assessment:
```