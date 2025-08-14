import openai
from utils.key import *

client = openai.OpenAI(api_key=openai_key)


def llm_generate_commentary(prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "system", "content": "You are a Football commentator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=20,
    )
    return response.choices[0].message.content.strip()


def build_pass_prompt(from_id, to_id):
    return f'Imagine you are Peter Drury. Now, Generate a 2-3 seconds of commentary for this short Football pass: Player {from_id} passes the ball to Player {to_id}.'
