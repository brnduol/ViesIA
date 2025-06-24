from flask import Flask, render_template, request, jsonify
import pandas as pd 
from funcoes.false_positive_rate import false_positive_rate


app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_csv():
    csv = request.get_data(as_text=True)
    df = pd.read_csv(next(request.files.values()))
    x = false_positive_rate(int(df['y_pred'].sum()), int(df['y_true'].sum()))
    
    return f'O False Positive Rate (FPR) é {x}'


    return csv
    


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=True)
 