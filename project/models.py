import pandas as pd
from markdown import markdown
from functions.helpers import bot_prompt


class BiasAnalysis:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        name_model: str,
        description: str,
        predictive_equality: float,
        spd: dict,
        fpr: float,
        disparate_impact: float,
    ) -> None:
        self.bot_notes = ''
        self.name_model = name_model
        self.description = description
        self.predictive_equality = predictive_equality
        self.spd = spd
        self.fpr = fpr
        self.disparate_impact = disparate_impact

    def add_bot_notes(self) -> None:
        prompt = f'''
        Predictive Equality: {self.predictive_equality};
        Statistical Parity Difference: {self.spd};
        False positive rate: {self.fpr};
        Disparate Impact: {self.disparate_impact};
        Descrição do problema: {self.description};
        '''
        self.bot_notes = markdown(bot_prompt(prompt))
