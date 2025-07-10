import pandas as pd

from functions.helpers import bot_prompt


# Note class
class Analysis:
    def __init__(
        self,
        name_model: str,
        description: str,
    ) -> None:
        self.name_model = name_model
        self.description = description
        self.dataframe = pd.read_csv('data/dataset.csv')
        self.bot_notes = None
        self.predictive_equality = None
        self.spd = None
        self.disparate_impact = None

    def add_bot_notes(self) -> None:
        prompt = f'''
        Predictive Equality: {self.predictive_equality};
        Statistical Parity Difference: {self.spd};
        Disparate Impact: {self.disparate_impact};
        Descrição do problema: {self.description};
        '''
        self.bot_notes = bot_prompt(prompt)
