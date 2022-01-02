"""
This script is used to plot the tag embeddings on validation set.
"""
import sys
sys.path.append('..')
import torch
import numpy as np
import os
os.chdir('../')
import json
import sklearn
from matplotlib import pyplot as plt

from encode import return_loaded_model
from models_t1000_att import TagMeanEncoder, TagSelfAttentionEncoder


def plot_tag_embeddings(tag_encoder, reshape_dim_2, plot_save_path):
    #tag_encoder = return_loaded_model(TagEncoder, tag_encoder_path)
    id2tag = json.load(open('./json/id2token_top_1000.json', 'rb'))
    tag_embeddings = []

    for tag_idx, _ in id2tag.items():
        tag_idx = int(tag_idx)
        tag_vector = torch.tensor(np.zeros(1000)).view(1, 1000).long()
        tag_vector[0, tag_idx] = 1
        embedding, _ = tag_encoder(tag_vector)
        tag_embeddings.append(embedding.tolist())

    data = np.array(tag_embeddings).reshape(1000, reshape_dim_2)
    tsne = sklearn.manifold.TSNE(n_components=2)
    tag_embeddings_tsne = tsne.fit_transform(data)    

    fig, ax = plt.subplots()
    for idx, (x, y) in enumerate(tag_embeddings_tsne):
        ax.scatter(x, y, alpha=0.8, c='red', edgecolors='none', s=5, marker="+")
        ax.annotate(id2tag[str(idx)], (x, y))

    #plt.show()
    plt.savefig(plot_save_path)

if __name__ == "__main__":
    for MODEL_NAME in [
        'ae_w2v_selfatt_c_1h/tag_encoder_att_epoch_200',
        'ae_w2v_128_selfatt_c_1h/tag_encoder_att_epoch_200',
        'ae_w2v_selfatt_c_4h/tag_encoder_att_epoch_200',
        'ae_w2v_128_selfatt_c_4h/tag_encoder_att_epoch_200',
        'ae_w2v_mean_c/tag_encoder_att_epoch_200',
        'ae_w2v_128_mean_c/tag_encoder_att_epoch_200',
    ]:
        MODEL_PATH = f'./saved_models/{MODEL_NAME}.pt'

        if 'ae_w2v_att_c' in MODEL_NAME:
            audio_model = return_loaded_model(
                AudioEncoder, './saved_models/ae_w2v_att_c_2/audio_encoder_epoch_200.pt'
            )
            audio_model.eval()
            model = return_loaded_model(
                lambda: TagAttentionEncoder(1001, 1152, 1, 1152, 1152, 1152),
                MODEL_PATH
            )
        elif 'ae_w2v_mean_c' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagMeanEncoder(1001, 1152, 1152),
                MODEL_PATH
            )
        elif 'ae_w2v_128_mean_c' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagMeanEncoder(1001, 128, 128),
                MODEL_PATH
            )
        elif 'ae_w2v_selfatt_c_1h' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagSelfAttentionEncoder(1001, 1152, 1, 1152, 1152, 1152),
                MODEL_PATH
            )
        elif 'ae_w2v_128_selfatt_c_1h' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagSelfAttentionEncoder(1001, 128, 1, 128, 128, 128),
                MODEL_PATH
            )
        elif 'ae_w2v_selfatt_c_4h' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagSelfAttentionEncoder(1001, 1152, 4, 1152, 1152, 1152),
                MODEL_PATH
            )
        elif 'ae_w2v_128_selfatt_c_4h' in MODEL_NAME:
            model = return_loaded_model(
                lambda: TagSelfAttentionEncoder(1001, 128, 4, 128, 128, 128),
                MODEL_PATH
            )

        model.eval()
        if '128' in MODEL_NAME:
            second_reshape_dimension = 128
        else:
            second_reshape_dimension = 1152
        PLOT_SAVE_PATH = 'plots/tag_embedding_'+MODEL_NAME.split("/")[0]+'.png'
        plot_tag_embeddings(model, second_reshape_dimension, PLOT_SAVE_PATH)
