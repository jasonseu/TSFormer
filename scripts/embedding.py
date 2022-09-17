# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: jasonseu
# Created on: 2022-3-22
# Email: zhuxuelin23@gmail.com
#
# Copyright © 2022 - CPSS Group
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
import argparse
import numpy as np
import torch
from transformers import BertTokenizer, BertModel

glove_path = '../SALRL_3/tmp/glove.840B.300d.txt'
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
token_mapping = {
    'diningtable': 'dining table',
    'pottedplant': 'potted plant',
    'tvmonitor': 'tv monitor'
}

def get_glove_embeddings(labels):
    with open(glove_path, 'r') as fr:
        embeddings = dict([line.split(' ', 1) for line in fr.readlines()])
        
    glove = []
    for label in labels:
        # category (eg: traffic light) with two or more words should split and average in each word embedding
        temp = np.array([list(map(lambda x: float(x), embeddings[t].split())) for t in label.split()])   
        if temp.shape[0] > 1:
            temp = temp.sum(axis=0, keepdims=True)
        glove.append(temp[0])
    glove = np.array(glove)
    return glove

def bert_text_preparation(text):
    marked_text = "[CLS] " + text
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1] * len(indexed_tokens)
    
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    return tokenized_text, tokens_tensor, segments_tensors

def get_bert_embeddings(tokens_tensor, segments_tensors):
    """Get embeddings from an embedding model
    
    Args:
        tokens_tensor (obj): Torch tensor size [n_tokens]
            with token ids for each token in text
        segments_tensors (obj): Torch tensor size [n_tokens]
            with segment ids for each token in text
        model (obj): Embedding model to generate embeddings
            from token and segment ids
    
    Returns:
        list: List of list of floats of size
            [n_tokens, n_embedding_dimensions]
            containing embeddings for each token
    
    """
    
    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings

def main(data):
    label_path = os.path.join('data', data, 'label.txt')
    labels = [t.strip() for t in open(label_path)]
    labels = [token_mapping[t] if t in token_mapping else t for t in labels]
    bert_embeddings = []
    for t in labels:
        _, tokens_tensor, segments_tensors = bert_text_preparation(t)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors)
        token_embedding = list_token_embeddings[1] if len(list_token_embeddings) == 2 else list_token_embeddings[0]
        bert_embeddings.append(token_embedding)
        # bert_embeddings.append(list_token_embeddings[0])
    bert_embeddings = np.array(bert_embeddings)
    np.save(os.path.join('data', data, 'bert.npy'), bert_embeddings)
    
    glove_embeddings = get_glove_embeddings(labels)
    np.save(os.path.join('data', data, 'glove.npy'), glove_embeddings)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='voc2007')
    args = parser.parse_args()
    main(args.data)