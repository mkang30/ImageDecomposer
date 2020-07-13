from fastapi import FastAPI, File
from starlette.responses import FileResponse
from maskrcnn import Segmentor
from zipfile import ZipFile
from PIL import Image
import io
import numpy as np


#model = get_segmentator()
model = Segmentor()

app = FastAPI(title="Image Decomposer",
              description='''Get the image decomposed into instances using the neural
              network model Mask R-CNN implemented in detectron2 library.''',
              version="0.1.0",
              )


@app.post("/segmentation")
def get_segmentation_map(file: bytes = File(...)):
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image = np.array(image)
    image = image[:,:,::-1].copy()
    output = model.predict(image)
    instances = model.decompose(image, output)

    with ZipFile('cont.zip', 'w') as zipObj:
        for key in instances:
            instances[key].save(key+".png")
            zipObj.write(key+".png")

    return FileResponse('cont.zip')
