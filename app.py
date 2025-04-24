from flask import Flask, request, render_template
import pickle
import numpy as np
from googletrans import Translator

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load scaler
with open('minmaxscaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Tạo translator
translator = Translator()

def translate_label(label):
    try:
        return translator.translate(label, dest='vi').text
    except Exception:
        return label  # Nếu lỗi, trả về nguyên gốc

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy dữ liệu từ form
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Dự đoán
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        # Dịch kết quả
        prediction_vi = translate_label(prediction)

        # Trả kết quả kèm hình
        image_path = f"img/{prediction}.jpg"
        return render_template('index.html',
                               prediction_text=f'Dự đoán: {prediction} ({prediction_vi})',
                               prediction_image=image_path)
    except Exception as e:
        return render_template('index.html', prediction_text=f'Lỗi: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
