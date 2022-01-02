"""
This script is used to compute mffc features for target task datasets.
Warning: Need manual editing for switching datasets
"""
import os
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
from pathlib import Path


#FILES_LOCATIONS = ['../data/UrbanSound8K/audio',\
#		'../data/GTZAN/genres',\
#		'../data/nsynth/nsynth-train/audio',\
#		'../data/nsynth/nsynth-test/audio']

SAVE_LOCATIONS = ['../data/embeddings/gtzan/mfcc/',\
                '../data/embeddings/us8k/mfcc/',\
		'../data/embeddings/nsynth/train/mfcc/',\
		'../data/embeddings/nsynth/test/mfcc/']

# GTZAN
p = Path('../data/GTZAN/genres')
filenames_gtzan = p.glob('**/*.wav')

# US8K
p = Path('../data/UrbanSound8K/audio')
filenames_us8k = p.glob('**/*.wav')

# NSynth
p = Path('../data/nsynth/nsynth-train/audio_selected')
filenames_nsynth_train = p.glob('*.wav')
p = Path('../data/nsynth/nsynth-test/audio')
filenames_nsynth_test = p.glob('*.wav')


dataset_files = [filenames_gtzan, filenames_us8k, filenames_nsynth_train, filenames_nsynth_test]
dataset_names = ['gtzan', 'us8k', 'nsynth/train', 'nsynth/test']


def compute_mfcc(filename, sr=22000):
    # zero pad and compute log mel spec
    try:
        audio, sr = librosa.load(filename, sr=sr, res_type='kaiser_fast')
    except:
        audio, o_sr = sf.read(filename)
        audio = librosa.core.resample(audio, o_sr, sr)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr)
    mfcc_delta = librosa.feature.delta(mfcc, width=5, mode='nearest')
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2, width=5, mode='nearest')
    
    feature = np.concatenate((np.mean(mfcc, axis=1), np.var(mfcc, axis=1),
                              np.mean(mfcc_delta, axis=1), np.var(mfcc_delta, axis=1),
                              np.mean(mfcc_delta2, axis=1), np.var(mfcc_delta2, axis=1)))

    return feature


if __name__ == "__main__":
    # p = Path(FILES_LOCATION)
    # filenames = p.glob('**/*.wav')
    # # filenames = p.glob('*')


    for filenames, ds_name, save_location in zip(dataset_files, dataset_names, SAVE_LOCATIONS):

        print(f'\n {ds_name}')
        #print(filenames)
        #print(save_location)
        for f in tqdm(filenames):
            #print(f)
            try:
	        #folder = f'./data/embeddings/{ds_name}/embeddings_{model_name}'
                Path(save_location).mkdir(parents=True, exist_ok=True)
                #print(save_location)
                y = compute_mfcc(str(f))
                np.save(Path(save_location, str(f.stem)+'.npy'), f)
            except Exception as e:
                print(e)
        print('\n')

#    for file_location, save_location in zip(FILES_LOCATIONS, SAVE_LOCATIONS):
#        p = Path(file_location)
#        filenames = p.glob('*.wav')
#        #print(filenames)
#        for f in tqdm(filenames):
#            try:
#                
#                y = compute_mfcc(str(f))
#                np.save(Path(save_location, str(f.stem)+'.npy'), y)
#            except RuntimeError as e:
#                print(e, f)
