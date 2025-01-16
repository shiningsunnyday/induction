from src.config import FILE_NAME, MODEL, MAX_SIZE, IMG_DIR, MAX_IMAGES
import io
import base64
import uuid
import os
# import openai
from PIL import Image
import numpy as np

# openai.api_key = os.getenv("api_key")


def encode(img_array):
    image = Image.fromarray(img_array)
    if (img_array.flatten() == 255).all():
        return None
    fname = f"{uuid.uuid4()}.png"
    image.save(os.path.join(IMG_DIR, fname))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue())
    base64_image = img_base64.decode("utf-8")
    os.remove(os.path.join(IMG_DIR, fname))
    return base64_image


def prepare_images(image_paths):
    base64_images = []
    images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            image = image_file.read()
            image = np.array(Image.open(image_path))
            # imgs = [image]
            # while imgs:
            #     image = imgs.pop(-1)
            #     if image.size > MAX_SIZE:
            #         if image.shape[0] > image.shape[1]:
            #             imgs.append(image[:image.shape[0]//2])
            #             imgs.append(image[image.shape[0]//2:])
            #         else:
            #             imgs.append(image[:,:image.shape[1]//2])
            #             imgs.append(image[:,image.shape[1]//2:])
            #     else:
            #         images.append(image)
            images.append(image)

    for image in images:
        image = encode(image)
        if image is not None:
            base64_images.append(image)
    return base64_images


def get_motifs(image_paths):
    # models = list(openai.Model.list()['data'])
    # print(sorted([m['id'] for m in models]))
    prompt = "".join(open(FILE_NAME).readlines())
    ans = ""
    for i in range((len(image_paths) + MAX_IMAGES - 1) // MAX_IMAGES):
        base64_images = prepare_images(
            image_paths[MAX_IMAGES * i : MAX_IMAGES * (i + 1)]
        )
        print(f"{len(base64_images)} patches")
        completion = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                    + [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        }
                        for base64_image in base64_images
                    ],
                }
            ],
        )
        # print(prompt)
        # print("=====PROMPT ABOVE, RESPONSE BELOW=====")
        res = completion.choices[0].message.content
        print(res)
        ans += f"{res}\n"
    return ans
