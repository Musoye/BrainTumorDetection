from flask import Flask, request, jsonify, render_template
import os
from predict import predict_mri
import uuid

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static'


@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html', result=[0, 1])
    elif request.method == 'POST':
        file = request.files['image']
        extension = file.filename.split('.')[-1]
        filename = str(uuid.uuid4()) + '.' + extension
        file.filename = filename
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(save_path)

        result = predict_mri(save_path)
        save_path = '../' + save_path
        return render_template('result.html', result=result, path=save_path)


if __name__ == '__main__':
    app.run(debug=True)