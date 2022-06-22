
from flask import Flask, request, render_template
from utils import clean_special, do_classification

app = Flask(__name__)


@app.route('/')
def home():
    initial_text = "Hello，this is COPD doctor. Send me your symptom and I will tell if you have gotten COPD." \
                   "The prediction would be much accurate if more information is given. " \
                   "Please keep each sentence within 25 words."
    return render_template('index.html', doctor_diagnosis=initial_text)


@app.route('/classify', methods=['POST'])
def classify():
    input_text = request.form['text']
    result = None
    if not input_text:
        result = "Please do not enter nothing."
    else:
        diagnosis_texts = clean_special(input_text)
        if len(diagnosis_texts) < 1:
            result = "Please enter more information."
        else:
            result = do_classification(diagnosis_texts)
            # json_output = do_NER(diagnosis_texts)
            # result += "\n" + json_output
    return render_template('index.html', doctor_diagnosis=result)


if __name__ == '__main__':
    app.run(debug=True)
