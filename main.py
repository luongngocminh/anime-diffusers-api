from diffusers import StableDiffusionPipeline
import torch
from flask import Flask, request, abort, send_file
import random
import string
import os

if not os.path.exists("./tmp"):
    os.makedirs("./tmp")


def randomword(length):
    return "".join(random.choice(string.ascii_lowercase) for i in range(length))


app = Flask(__name__)


@app.route("/gen", methods=["POST"])
def generate():
    if pipe:
        try:
            print(request.json)
            random_name = randomword(20)
            prompt = request.json["prompt"]
            print(prompt)
            image = pipe(prompt).images[0]
            print(image)
            image_name = "./tmp/" + random_name + ".png"
            image.save(image_name)
            return send_file(image_name)

        except Exception as e:
            print(e)
            abort(500)


if __name__ == "__main__":
    port = 12345  # If you don't provide any port the port will be set to 12345

    model_id = "andite/anything-v4.0"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16)
    pipe = pipe.to("mps")
    app.run(port=port, debug=True)
