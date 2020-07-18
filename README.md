# ImageDecomposer
ImageDecomposer is an e2e app that detects objects in an input image and outputs them as separate images. Backend is constructed using `FastAPI`, frontend - `streamlit`. The underlying mechanism involves Mask R-CNN implemented in `detectron2` library.
# Description
Input an image

<img src="https://github.com/mkang30/ImageDecomposer/blob/master/sci.jpg" width="300" height="350"/>

And the program outputs segmented instances as separate images

<img src="https://github.com/mkang30/ImageDecomposer/blob/master/idsc1.png" width="480" height="600"/>

# Running

To run the app on your own device clone the repository, unzip fastapi/detectron2. Then in machine with Docker run commands:
    
    docker-compose build
    docker-compose up

Visit http://localhost:8501. 
