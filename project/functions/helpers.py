import os
from flask import Flask
from openai import OpenAI
from markdown import markdown
from dotenv import load_dotenv

load_dotenv('project/.env')

ALLOWED_EXTENSIONS = {'csv'}

client = OpenAI(api_key=os.getenv('OPEN_AI_KEY'), base_url='https://api.perplexity.ai')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def bot_prompt(prompt: str) -> str:
    '''
    Takes a string as prompt and returns a response generated based
    on said prompt.
    '''

    response = client.chat.completions.create(
        model='sonar-pro',
        response_format={'type': 'text'},
        messages=[
            {
                'role': 'system',
                'content': '''
                            Você é um especialista em Análise de viés e Markdown.
                            
                            A partir de scores de:
                            
                            - Predictive Equality;
                            - Statistical Parity Difference;
                            - e Disparate Impact;
                            - False Positive Rate;
                            
                            além de uma descrição do problema, descreva o que pode ser feito de modo a mitigar o viés do modelo.
                            
                            Não forneça exemplos de código, apenas utilize noções teóricas.
                            ''',
            },
            {'role': 'user', 'content': prompt},
        ],
    )

    answer = response.choices[0].message.content

    if answer is None:
        return ''
    return answer
