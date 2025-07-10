import os
import pandas as pd
from flask import Flask, request, redirect, url_for, render_template, flash

from functions.helpers import *
from functions.metrics import false_positive_rate, disparate_impact

app = Flask(__name__)
app.secret_key = os.getenv('APP_SECRET_KEY')


@app.route('/upload', methods=['POST'])
def upload_csv():
    uploaded_file = next(iter(request.files.values()))
    df = pd.read_csv(uploaded_file.stream)
    # Tem que codificar as features (por exemplo masculino e feminino)
    x = false_positive_rate(df)
    di = disparate_impact(df, 'atributo_sensivel', 'masculino')
    print(f'O Disparate Impact (DI) é {di}')

    return f'O False Positive Rate (FPR) é {x}'


@app.route('/forms', methods=['GET', 'POST'])
def forms():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')
            file.save(filepath)
            return redirect(url_for('forms'))
        else:
            flash('Arquivo inválido. Envie um arquivo .csv.')
            return redirect(request.url)
    return render_template('your_form_template.html')


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
