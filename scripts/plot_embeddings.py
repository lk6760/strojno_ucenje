"""
This script is used to plot the audio and tag based embeddings for the clips 
from the validation set.
"""
import sys
sys.path.append('..')
import torch
import numpy as np
import sklearn
import pickle
import os
os.chdir('../')
import json
from torch.utils import data
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from tqdm import tqdm


from data_loader import HDF5Dataset
from encode import extract_tag_embedding, extract_audio_embedding, return_loaded_model
from models_t1000_att import AudioEncoder, TagMeanEncoder, TagSelfAttentionEncoder


TAG_MODEL_NAMES = [
#            'ae_w2v_128_selfatt_c_1h/tag_encoder_att_epoch_200',
#            'ae_w2v_128_selfatt_c_4h/tag_encoder_att_epoch_200',
#            'ae_w2v_selfatt_c_1h/tag_encoder_att_epoch_200',
#            'ae_w2v_selfatt_c_4h/tag_encoder_att_epoch_200',
#            'ae_w2v_128_mean_c/tag_encoder_att_epoch_200',
#            'ae_w2v_mean_c/tag_encoder_att_epoch_200',
             'ae_w2v_selfatt_c_4h_new/tag_encoder_att_epoch_20',
             'ae_w2v_selfatt_c_4h_new/tag_encoder_att_epoch_100',
             'ae_w2v_selfatt_c_4h_new/tag_encoder_att_epoch_200',
                  ]

AUDIO_MODEL_NAMES = [
#           'ae_w2v_128_selfatt_c_1h/audio_encoder_epoch_200',
#           'ae_w2v_128_selfatt_c_4h/audio_encoder_epoch_200',
#           'ae_w2v_selfatt_c_1h/audio_encoder_epoch_200',
#           'ae_w2v_selfatt_c_4h/audio_encoder_epoch_200',
#           'ae_w2v_128_mean_c/audio_encoder_epoch_200',
#           'ae_w2v_mean_c/audio_encoder_epoch_200',
            'ae_w2v_selfatt_c_4h_new/audio_encoder_epoch_20',
            'ae_w2v_selfatt_c_4h_new/audio_encoder_epoch_100',
            'ae_w2v_selfatt_c_4h_new/audio_encoder_epoch_200',
                    ]

N = 1000

loader_params = {'batch_size': 1, 'shuffle': True, 'num_workers': 1}
dataset_val = HDF5Dataset('./hdf5_ds/spec_tags_top_1000_val')
test_loader = data.DataLoader(dataset_val, **loader_params)


def extract_audio_tag_embeddings(audio_encoder,tag_encoder, plot_path, filenames, dataset):
    #sound_id2idx = {id:idx for idx, id in 
    #                enumerate(list(dataset_val.h_file['dataset']['id'][:,0]))}

    #audio_encoder = return_loaded_model(AudioEncoder, '../saved_models/dual_ae_c/audio_encoder_epoch_200.pt')
    #tag_encoder = return_loaded_model(TagEncoder, '../saved_models/dual_ae_c/tag_encoder_epoch_200.pt')

    audio_embeddings = []
    tag_embeddings = []
    #sound_ids = []
    genre_list = []

    if dataset == 'spotify':
        split_symbol = "_"
    else:
        split_symbol = "."
    #print(split_symbol)
    #for idx, (data, tags, sound_id) in enumerate(test_loader):
    for idx, f in enumerate(tqdm(filenames)):
        #sound_ids += sound_id.tolist()
        #x = data.view(-1, 1, 96, 96).clamp(0)
        #tags = tags.long().clamp(0)
        #print(x.shape, tags.shape)
        # encode
        #z_audio, z_d_audio = audio_encoder(x)
        #z_tags, z_d_tags = tag_encoder(tags)
        #print(idx)
        #z_audio = np.load(str(f))
        #print(str(f.stem))
        z_audio, _ = extract_audio_embedding(audio_encoder, str(f))
        audio_embeddings.append(z_audio.tolist())
        #tag_embeddings.append(z_tags.tolist())
        #print(str(f.stem))
        genre = str(f.stem).split(split_symbol)[0]
        if genre not in genre_list:
            genre_list.append(genre)
        if idx == N:
            #print(len(genre_list))
            break

    #print(z_audio.shape, z_tags.shape)
    size_embedding = z_audio.shape[1]
    audio_embeddings = np.array(audio_embeddings).reshape(len(audio_embeddings), size_embedding)[:N, :]
    #tag_embeddings = np.array(tag_embeddings).reshape(len(tag_embeddings), size_embedding)[:N, :]

    #data = np.concatenate((audio_embeddings, tag_embeddings), 0)
    data = np.array(audio_embeddings)
    tsne = sklearn.manifold.TSNE(n_components=2)
    data = tsne.fit_transform(data)
    audio_embeddings_tsne = data[:N, :]
    #tag_embeddings_tsne = data[N:, :]
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    color_list = ['red', 'blue', 'darkgreen', 'orange', 'grey', 
                'lime', 'magenta','cyan', 'purple', 'black']
    

    for idx, (x, y) in enumerate(audio_embeddings_tsne):
        color = color_list[(idx//100)]
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=8)
        # ax.annotate(sound_ids[idx][0], (x, y))

    #for idx, (x, y) in enumerate(tag_embeddings_tsne):
    #    ax.scatter(x, y, alpha=0.8, c='blue', edgecolors='none', s=8)
        # ax.annotate(sound_ids[idx][0], (x, y))

    #for (x0,y0), (x1,y1) in zip(audio_embeddings_tsne, tag_embeddings_tsne):
    #    ax.plot((x0,x1), (y0,y1), linewidth=0.2, color='black', alpha=0.6)

    plt.title('Visualisation of the aligned learnt representations (TSNE)')
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=genre_list[0],
               markerfacecolor=color_list[0], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[1],
               markerfacecolor=color_list[1], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[2],
               markerfacecolor=color_list[2], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[3],
               markerfacecolor=color_list[3], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[4],
               markerfacecolor=color_list[4], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[5],
               markerfacecolor=color_list[5], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[6],
               markerfacecolor=color_list[6], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[7],
               markerfacecolor=color_list[7], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[8],
               markerfacecolor=color_list[8], markersize=4),
        Line2D([0], [0], marker='o', color='w', label=genre_list[9],
               markerfacecolor=color_list[9], markersize=4),
    ]  
    plt.legend(handles=legend_elements, loc=2)
    plt.savefig(plot_path)
    #plt.show()
    

if __name__ == "__main__":
    for AUDIO_MODEL_NAME, MODEL_NAME in zip(AUDIO_MODEL_NAMES, TAG_MODEL_NAMES):
        AUDIO_MODEL_PATH = f'./saved_models/{AUDIO_MODEL_NAME}.pt'

        if 'cnn' in AUDIO_MODEL_NAME:
            model = return_loaded_model(CNN,AUDIO_MODEL_PATH)
        elif 'w2v' in AUDIO_MODEL_NAME:
            if '128' in AUDIO_MODEL_NAME:
                s = 128
            else:
                s = 1152
            #print(s)
            audio_model = return_loaded_model(lambda: AudioEncoder(s), AUDIO_MODEL_PATH)
        else:
            audio_model = return_loaded_model(AudioEncoder, AUDIO_MODEL_PATH)

        audio_model.eval()

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
        #p = Path('./data/embeddings/gtzan/embeddings_ae_w2v_selfatt_c_4h_200')
        #filenames = p.glob('*.npy')
        p = Path('./data/Spotify_GTZAN_genres')
        spotify_filenames = sorted(p.glob('**/*.wav'))
        p = Path('./data/GTZAN/genres')
        gtzan_filenames = sorted(p.glob('**/*.wav'))

        for i, filenames in enumerate([spotify_filenames, gtzan_filenames]):
            if i == 0:
                ds_name = 'spotify'
            else:
                ds_name = 'gtzan'
            #print(ds_name)
            PLOT_SAVE_PATH = 'plots/'+ds_name+'_embedding_'+MODEL_NAME.split('/')[0]+'_'+MODEL_NAME.split('/')[1]+'.png'
            #print(MODEL_NAME, AUDIO_MODEL_NAME)
            extract_audio_tag_embeddings(audio_model, model, PLOT_SAVE_PATH, filenames, ds_name)
