from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")
print("✅ Model Loaded:", type(model))


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # 👇 Fixed 29 features (V1–V28 + Amount)
        features = [
            float(data.get('V1', 0)), float(data.get('V2', 0)), float(data.get('V3', 0)),
            float(data.get('V4', 0)), float(data.get('V5', 0)), float(data.get('V6', 0)),
            float(data.get('V7', 0)), float(data.get('V8', 0)), float(data.get('V9', 0)),
            float(data.get('V10', 0)), float(data.get('V11', 0)), float(data.get('V12', 0)),
            float(data.get('V13', 0)), float(data.get('V14', 0)), float(data.get('V15', 0)),
            float(data.get('V16', 0)), float(data.get('V17', 0)), float(data.get('V18', 0)),
            float(data.get('V19', 0)), float(data.get('V20', 0)), float(data.get('V21', 0)),
            float(data.get('V22', 0)), float(data.get('V23', 0)), float(data.get('V24', 0)),
            float(data.get('V25', 0)), float(data.get('V26', 0)), float(data.get('V27', 0)),
            float(data.get('V28', 0)), float(data.get('Amount', 0))
        ]

        # 🔍 Debug prints (VERY IMPORTANT)
        print("📊 Feature Length:", len(features))     # must be 29
        print("📊 Features:", features)

        # Convert to numpy array (correct shape)
        final_features = np.array(features).reshape(1, -1)
        print("📊 Shape:", final_features.shape)       # must be (1,29)

        # Prediction
        prediction = model.predict(final_features)
        print("📊 Raw Prediction:", prediction)

        # Safety check
        if prediction is None or len(prediction) == 0:
            return "❌ Error: Model returned empty prediction"

        # Output
        result = "⚠️ Fraud Transaction" if prediction[0] == 1 else "✅ Legit Transaction"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return f"❌ Error: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)