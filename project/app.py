import os
import pandas as pd
from flask import Flask, request, redirect, url_for, render_template, flash

from models import BiasAnalysis
from functions.metrics import *
from functions.helpers import allowed_file

app = Flask(__name__)
app.secret_key = os.getenv('APP_SECRET_KEY')
app.config['UPLOAD_FOLDER'] = './data'


@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return render_template('forms.html')
    return render_template('home.html')


@app.route('/forms', methods=['GET', 'POST'])
def forms():
    if request.method == 'POST':
        name = request.form['name']
        problem: str = request.form['problem']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')

            upload_dir = os.path.dirname(filepath)  # 'data'

            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file.save(filepath)

            analysis = BiasAnalysis(name_model=name, description=problem)
            analysis.predictive_equality = predictive_equality_fpr_diff(
                df=analysis.dataframe,
                privileged_value=1,
                unprivileged_value=0,
            )
            analysis.spd = statistical_parity_difference(
                df=analysis.dataframe,
                privileged_value=1,
            )
            analysis.fpr = false_positive_rate(analysis.dataframe)
            analysis.disparate_impact = disparate_impact(
                df=analysis.dataframe,
                sensitive_attr='sensitive_attr',
                privileged_value=1,
            )
            analysis.add_bot_notes()

            return redirect(url_for('results'))
        else:
            flash('Arquivo inválido. Envie um arquivo .csv.')
            return redirect(request.url)

    return render_template('forms.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    analysis = BiasAnalysis()
    return render_template('results.html', analysis=analysis)


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
