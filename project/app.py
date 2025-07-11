import os
import pandas as pd
from flask import Flask, request, redirect, url_for, render_template, flash

from models import BiasAnalysis
from functions.metrics import predictive_equality_fpr_diff, false_positive_rate
from functions.metrics import statistical_parity_difference, disparate_impact
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
        problem = request.form['problem']
        privileged_value = request.form['privileged_value']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'dataset.csv')

            upload_dir = os.path.dirname(filepath)  # 'data'

            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            file.save(filepath)

            dataframe = pd.read_csv(filepath)

            predictive_equality = predictive_equality_fpr_diff(
                df=dataframe,
                privileged_value=privileged_value,
            )
            spd = statistical_parity_difference(
                df=dataframe,
                privileged_value=privileged_value,
            )
            di = disparate_impact(
                df=dataframe,
                sensitive_attr='sensitive_attr',
                privileged_value=privileged_value,
            )
            fpr = false_positive_rate(dataframe)
            analysis = BiasAnalysis(
                name_model=name,
                description=problem,
                predictive_equality=predictive_equality,
                spd=spd,
                fpr=fpr,
                disparate_impact=di,
            )
            analysis.add_bot_notes()

            return redirect(url_for('results'))
        else:
            flash('Arquivo inválido. Envie um arquivo .csv.')
            return redirect(request.url)

    return render_template('forms.html')


@app.route('/results', methods=['GET', 'POST'])
def results():
    analysis = BiasAnalysis._instance
    return render_template('results.html', analysis=analysis)


if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
