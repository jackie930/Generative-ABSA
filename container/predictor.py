# -*- coding: utf-8 -*-

import os
import json
import boto3

import datetime
import warnings
import numpy as np
import torch
from absa import T5FineTuner
import time
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5ForConditionalGeneration, 
    T5Tokenizer,
    set_seed,
)
warnings.filterwarnings("ignore",category=FutureWarning)

import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import codecs
import flask

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

print ("loading pretrained models!")
# Set up model
n_gpu=0
max_seq_length=512
device = torch.device(f'cuda:{n_gpu}')

saved_model_dir = 'opt/ml/model'
all_checkpoints = []
for f in os.listdir(saved_model_dir):
    file_name = os.path.join(saved_model_dir, f)
    if 'cktepoch' in file_name:
        all_checkpoints.append(file_name)
print ("all checkpoints: ", all_checkpoints)

checkpoint = os.path.join(saved_model_dir,all_checkpoints[-1])

model_ckpt = torch.load(checkpoint, map_location=device)
model = T5FineTuner(model_ckpt['hyper_parameters'])
model.load_state_dict(model_ckpt['state_dict'])
tokenizer = T5Tokenizer.from_pretrained('t5-base')

model.model.to(device)
model.model.eval()
# tokenizer = AutoTokenizer.from_pretrained('model')
# model = AutoModelForSeq2SeqLM.from_pretrained('model')

# model.resize_token_embeddings(len(tokenizer))
print ("<<<< loading pretrained models success")

def absa_infer(data):
    """do predict"""
    t1 = datetime.datetime.now()
    inputs = tokenizer(
              data, max_length=max_seq_length, pad_to_max_length=True, truncation=True,
              return_tensors="pt",
            )
    outs = model.model.generate(input_ids=inputs["input_ids"].to(device), 
                                    attention_mask=inputs["attention_mask"].to(device), 
                                    max_length=1024)
#     print(outs[0])
    dec=tokenizer.decode(outs[0], skip_special_tokens=True)
    print("<<<<done")
    print("prediction result: ",dec)
    t2 = datetime.datetime.now()
    return dec ,str(t2 - t1)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    tic = time.time()
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")
    data = flask.request.data.decode('utf-8')
    print("<<<<<input data: ", data)
    
    data = json.loads(data)
    data_input = data['data']

    #inference
    res, infer_time = absa_infer(data_input)
    
    print("Done inference! ")
    print("res: ", res)
    inference_result = {
        'result': res,
        'infer_time': infer_time
    }
    
    _payload = json.dumps(inference_result, ensure_ascii=False)
    return flask.Response(response=_payload, status=200, mimetype='application/json')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    
    