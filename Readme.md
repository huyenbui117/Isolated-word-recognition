# Speech Processing project
 _19021307 Bui Khanh Huyen_
 
[![Python Version](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/) [![Python Version](https://img.shields.io/badge/FastAPI-0.75.0-blue.svg)](https://fastapi.tiangolo.com/)
Isolated word recognition using Dynamic Time Warping and Hidden Markov model

## Features

- Preprocess raw dataset by segmentation using preprocess.py
- Extract features (mfcc, $$\Delta_{mfcc}, \Delta \Delta_{mfcc}^{}$$) using feature_extraction.py

- Predict isolated word by dtw using dtw.py
- Predict isolated word by hmm using hmm_model.py
- Predict your own word using api provided by inference.py

## Installation

- Clone the code, intall required packages

```shell
git clone https://github.com/huyenbui117/SP.git
pip install  -r requirements.txt
```
## Evaluate
```shell
py dtw.py
```
![dtw](https://github.com/huyenbui117/SP/blob/master/gifs/dtw.gif)
```shell
py hmm_model.py
```
![dtw](https://github.com/huyenbui117/SP/blob/master/gifs/hmm_model.png)
## Inference
- Start the server:

```shell
uvicorn reference:app --reload
```

- Go to [localhost:8000/docs](http://localhost:8000/docs), click `POST` &rarr; `Try it out` and try to upload data as
  .`wav` file
![start_the_server](https://github.com/huyenbui117/SP/blob/master/gifs/start_the_server.gif)
- Click `Execute` to get results
![api](https://github.com/huyenbui117/SP/blob/master/gifs/api.gif)
