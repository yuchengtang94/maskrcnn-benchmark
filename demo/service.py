import requests
from io import BytesIO
from PIL import Image
import numpy as np
import io

import flask
from gevent import monkey
from gevent import pywsgi
monkey.patch_all()

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

config_file = "../configs/text/e2e_faster_rcnn_X_101_32x8d_FPN_pooler_lr003.yaml"
# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

app = flask.Flask(__name__)

coco_demo = COCODemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.5,
)

def predict_result(img):
    pil_image = Image.open(img).convert('RGB')
    image = np.array(pil_image)[:, :, [2, 1, 0]]
    predictions = coco_demo.compute_prediction(image)
    top_predictions = coco_demo.select_top_predictions(predictions)
    return top_predictions

@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    data = {"success": False}
    
    # Ensure an image was properly uploaded to our endpoint.
    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            # Read the image in PIL format
            image = flask.request.files["image"].read()
            
            # Preprocess the image and prepare it for classification.
            top_predictions = predict_result(io.BytesIO(image))
            data['bbox'] = top_predictions.bbox.tolist()
            data['labels'] = top_predictions.get_field('labels').tolist()
            data["success"] = True
            
            
    # Return the data dictionary as a JSON response.
    return flask.jsonify(data)

if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8000), app)
    server.serve_forever()