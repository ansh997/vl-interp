import os
import json
import torch

from methods.algorithms import internal_confidence_heatmap
from methods.llava_utils import retrieve_logit_lens_llava, load_llava_state

model_state = load_llava_state()
retrieve_logit_lens = retrieve_logit_lens_llava

directory = '/root/vl-interp/images/cifar10_images'
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
file_path = './data.json'


for img in os.listdir(directory):
    img_path = os.path.join(directory, img)
    print('Begin processing image:', img)
    caption, softmax_probs = retrieve_logit_lens(model_state, img_path)
    final_probs = []
    for class_ in classes:
        heatmap_data = internal_confidence_heatmap(model_state['tokenizer'], softmax_probs, class_).tolist()
        final_probs.append(heatmap_data)

    with open(file_path, 'r') as f:
        data = json.load(f)
    data[img] = final_probs
    with open(file_path, 'w') as f:
        json.dump(data, f)
    torch.cuda.empty_cache()
    print('End of processing image:', img)