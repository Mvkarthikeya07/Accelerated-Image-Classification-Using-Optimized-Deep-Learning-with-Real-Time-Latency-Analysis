from flask import Flask, render_template, request
import os
from client.inference import predict_image

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    latency = None
    image_path = None

    if request.method == "POST":
        image = request.files["image"]

        # Save image inside static folder
        image_path = os.path.join(app.config["UPLOAD_FOLDER"], image.filename)
        image.save(image_path)

        prediction, confidence, latency = predict_image(image_path)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        latency=latency,
        image_path=image_path
    )

if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    app.run(debug=True)
