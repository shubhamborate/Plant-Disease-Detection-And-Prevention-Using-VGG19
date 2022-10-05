from flask import Flask, render_template, request, jsonify
import os
from PIL import Image
from predict import prediction, getDataFromCSV

DEVELOPMENT_ENV = False

app = Flask(__name__)

if not os.path.isdir(os.path.join(os.getcwd(), 'images')):
    os.mkdir(os.path.join(os.getcwd(), 'images'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/result')
def about():
    product_id = request.args.get('id', default=-1, type=int)

    
    app_data = {
        "disease_name": "undefined",
        "supplement name": "null",
        "supplement image": "null",
        "buy link": "null",
        
    }
    if product_id != -1:
        dataPredicted = getDataFromCSV(product_id)
        if any(dataPredicted):
            app_data = {
                "disease_name": dataPredicted[1],
                "supplement name": dataPredicted[2],
                "supplement image": dataPredicted[3],
                "buy link": dataPredicted[4],
                
            }

    
    return render_template('result.html', app_data=app_data)


@app.route('/analyze', methods=['POST'])
def analyze():
    image = request.files['file']
    
    pathOfFile = os.path.join(os.getcwd(), 'images', image.filename)
    image.save(pathOfFile)
    data = {}

    
    data['product_id'] = prediction(pathOfFile)
    os.remove(pathOfFile)
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
