from src.config import FILE_NAME, MODEL
import base64
import os
import openai
openai.api_key = os.getenv('api_key')

def get_motifs(image_paths):
    models = list(openai.Model.list()['data'])
    print(sorted([m['id'] for m in models]))
    prompt = ''.join(open(FILE_NAME).readlines())    
    base64_images = []
    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')    
            base64_images.append(base64_image)
    completion = openai.ChatCompletion.create(model=MODEL, 
                                            messages=[{"role": "user", 
                                                     "content": [
                                                         {"type": "text",
                                                          "text": prompt}]+[
                                                              {"type": "image_url",
                                                               "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                                               } for base64_image in base64_images
                                                          ]}],
                                       )
    print(prompt)
    print("=====PROMPT ABOVE, RESPONSE BELOW=====")
    res = completion.choices[0].message.content
    print(res)
    return res
