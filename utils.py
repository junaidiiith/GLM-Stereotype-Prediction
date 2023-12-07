import model2nx
from collections import deque
import datetime
import os
import fnmatch
import random
import numpy as np
from torch.utils import tensorboard
import xmltodict
import json
import torch.nn.functional as F
import torch
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
   


EOP = " <|endofpath|> "
STP = " <|stereotype|> "
CLASS = " <|class|> "
REL = " <|relation|> "
TYPE = " <|type|> "
ATTR = " <|attributes|> "

NODE = ' <|node|> '
EDGE = ' <|edge|> '

BOS = '<|startoftext|>'
EOS = eos_token='<|endoftext|>'
PAD = pad_token='<|pad|>'

m2t = {
    "Class": CLASS,
    "Relation": REL,
}

ontouml_keywords = [EOP, STP, CLASS, REL, TYPE, ATTR, NODE, EDGE, BOS, EOS]
ontouml_meta_properties = model2nx.extra_properties

# Function to perform BFS on JSON
TYPE_NODE = '@xsi:type'
NAME = '@name'
SEP = " ==> "


def xml_to_json(xml_string):
    xml_dict = xmltodict.parse(xml_string)
    json_data = json.dumps(xml_dict, indent=4)
    return json_data


def merge_dicts(d1, d2):
    for k in d2:
        prev_v = d1.get(k, (0, 0))
        new_v = (prev_v[0] + d2[k], prev_v[1] + 1)
        d1[k] = new_v



def find_files_with_extension(root_dir, extension):
    matching_files = []

    # Recursively search for files with the specified extension
    for root, _, files in os.walk(root_dir):
        for filename in fnmatch.filter(files, f'*.{extension}'):
            matching_files.append(os.path.join(root, filename))

    return matching_files


def set_config(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model_name = args.model_name.replace('/', '_')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_file_name = f'{args.results_file_name}_{model_name}-{current_time}.txt'
    logs_dir = args.log_dir + f'/{current_time}'
    os.makedirs(logs_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(log_dir=logs_dir)

    trained_models_dir = args.trained_models
    os.makedirs(trained_models_dir, exist_ok=True)
    best_model_name = f'{trained_models_dir}/best_model-{model_name}{current_time}.pth'

    with open(f"{results_file_name}", 'a') as f:
        f.write(f"{args}\n")
    
    return results_file_name, writer, best_model_name


def print_cuda_memory_allocation():
    allocated_memory = torch.cuda.memory_allocated()

    # Get the current GPU memory managed by the caching allocator in bytes
    cached_memory = torch.cuda.memory_cached()

    print(f"Allocated GPU memory: {allocated_memory / (1024**2):.2f} MB")
    print(f"Cached GPU memory: {cached_memory / (1024**2):.2f} MB")


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))


def get_free_and_available_memory():
    # Get the current GPU memory managed by the caching allocator in bytes
    cached_memory = torch.cuda.memory_cached()
    allocated_memory = torch.cuda.memory_allocated()
    free_memory = cached_memory - allocated_memory
    print(f"Free GPU memory: {free_memory / (1024**2):.2f} MB")
    print(f"Allocated GPU memory: {allocated_memory / (1024**2):.2f} MB")
    


def summary_reader(log_dir):
    def my_summary_iterator(path):
        for r in tf_record.tf_record_iterator(path):
            yield event_pb2.Event.FromString(r)

    for filename in os.listdir(log_dir):
        path = os.path.join(log_dir, filename)
        for event in my_summary_iterator(path):
            for value in event.summary.value:
                print(value.tag, event.step, value.tensor)


def format_time(elapsed):
   return str(datetime.timedelta(seconds=int(round((elapsed)))))




def test_model(model, shape):
    ## Load Automodel
    input_ids = torch.randint(0, 1000, shape).to('cuda')
    attention_mask = torch.ones(shape).to('cuda')
    output = model(input_ids, attention_mask=attention_mask)
    print(output[0].shape)
