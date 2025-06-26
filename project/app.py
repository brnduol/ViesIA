from flask import Flask, request
import pandas as pd
from functions.metrics import false_positive_rate, disparate_impact

app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_csv():
    uploaded_file = next(iter(request.files.values()))
    df = pd.read_csv(uploaded_file.stream)
    # Tem que codificar as features (por exemplo masculino e feminino)
    x = false_positive_rate(df)
    di = disparate_impact(df, 'atributo_sensivel', 'masculino')
    print(f'O Disparate Impact (DI) é {di}')

    return f'O False Positive Rate (FPR) é {x}'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
