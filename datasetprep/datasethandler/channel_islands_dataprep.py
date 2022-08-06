# %%
import json
from itertools import groupby
from operator import itemgetter
import random
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)
np.random.seed(42)


class config: 
    save_pickles = False
    run_eda = True
    metadata_path = "C:\Projects\wild\data\islands\metadata.json"
    pickle_path = 'C:\Projects\seq2bbox\data\pickle\islands\\'

with open(config.metadata_path) as f:
    d = json.load(f)


# some basic data prep..


human_imageid = [i.get('image_id') for i in d['annotations'] if i.get('category_id') == 1] #5981 human labeled. 

# filter out all images that are in a sequence shorter than 3
noseq_imageid = [i.get('id') for i in d['images'] if i.get('seq_num_frames') <= 2]
id_filter = set(noseq_imageid + human_imageid)
del human_imageid
del noseq_imageid
d['annotations'] = [i for i in d['annotations'] if i.get('image_id') not in id_filter]
d['images'] = [i for i in d['images'] if i.get('id') not in id_filter]

# change the category id of 'other' to that of human
print('Changing category id of "other" to that of human...')

for anno in d['annotations']:
    if anno.get('category_id') == 6:
        anno['category_id'] = 1


# merging the annotations and image metadata
for i in range(len(d['images'])):
    d['images'][i]['image_id'] = d['images'][i].pop('id')

my_id = itemgetter('image_id')
meta_anno = []

for k, v in groupby(sorted((d['annotations'] + d['images']), key=my_id), key=my_id):
    meta_anno.append({key:val for d in v for key, val in d.items()})


# this logic helps with stratification by all categories in the sequence...
def get_ordered_sequence_categories(dataset, all_sequences):
    imgs_seq_lookup = {}
    for ma in dataset:
        imgs_seq_lookup.setdefault(ma.get('seq_id','empty'),[]).append(ma)
    orderedseqcats = []
    for seq in all_sequences: 
        cats = [i.get('category_id') for i in imgs_seq_lookup.get(seq)]

        # the following logic finds the most frequent category in the sequence, not including 'empty'. 
        catsnozero = [c for c in cats if c != 0]
        if not catsnozero:  
            maxcats = 0
        else: 
             maxcats = max(set(catsnozero), key=catsnozero.count)

        orderedseqcats.append(maxcats)
    return orderedseqcats

def sequences_to_image_annotations(sequences, annotations):
    sequences = set(sequences)
    images_anns = []
    for ann in annotations:
        if ann.get('seq_id') in sequences:
            images_anns.append(ann)
    return images_anns

def stratified_sequence_split(annotations, partition):
    all_sequences = sorted(list(set([i.get('seq_id') for i in annotations])))
    full_length = len(all_sequences)
    orderedseqcats = get_ordered_sequence_categories(meta_anno,all_sequences)
    labels = [i for i in orderedseqcats]
    large_seq, small_seq = train_test_split(all_sequences, test_size=(full_length//partition), random_state=42, stratify=labels)
    large_anno, small_anno = sequences_to_image_annotations(large_seq, annotations), sequences_to_image_annotations(small_seq, annotations) 
    return large_anno, small_anno

train100, test = stratified_sequence_split(meta_anno, 5)
train100, val =  stratified_sequence_split(train100, 5)
_, test20  = stratified_sequence_split(test, 5)
_, val20 =  stratified_sequence_split(val, 5)
_, train50 = stratified_sequence_split(train100, 2)
_, train20 = stratified_sequence_split(train100, 5)
_, train10 = stratified_sequence_split(train100, 10)
_, train5 = stratified_sequence_split(train100, 20)

if config.save_pickles:
    for name, dset in zip(['test', 'val', 'train100', 'test20', 'val20', 'train50', 'train20', 'train10', 'train5'], [test, val, train100, test20, val20, train50, train20, train10, train5]):
        with open(config.pickle_path + name + '.pk', 'wb') as f:
            pickle.dump(dset, f, protocol=pickle.HIGHEST_PROTOCOL)


###
### EDA & Thesis Plotting
### 
import sys
if not config.run_eda:
    sys.exit('Finished running, exiting...')


#%% 
# EDA
def show_values(axs, orient="v", space=.01):
    def _single(ax):
        if orient == "v":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
                value = '{:.0f}'.format(p.get_height())
                ax.text(_x, _y, value, ha="center") 
        elif orient == "h":
            for p in ax.patches:
                _x = p.get_x() + p.get_width() + float(space)
                _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
                value = '{:.0f}'.format(p.get_width())
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


for setname, dataset in zip(['test', 'val', 'train100', 'test20', 'val20', 'train50', 'train20', 'train10', 'train5'], [test, val, train100, test20, val20, train50, train20, train10, train5]):
    location_counter = {}
    sequence_counter = {}
    species_counter = {}

    for img in dataset:
        # location_counter[img.get('location')] = location_counter.get(img.get('location'), 0) + 1
        # sequence_counter[img.get('seq_num_frames')] = sequence_counter.get(img.get('seq_num_frames'), 0) + 1
        species_counter[img.get('category_id')] = species_counter.get(img.get('category_id'), 0) + 1

    species_lookup = {i.get('id') : i.get('name') for i in d['categories']}

    # replace humans by other in lookup:
    species_lookup[1] = 'other'


    species_named_counter = {species_lookup.get(k): v for k, v in species_counter.items()}

    plt.figure(num=None, figsize=(10,8), dpi=80, facecolor='w', edgecolor='r')
    y = list(species_named_counter.keys())
    x = list(species_named_counter.values())
    argx = np.argsort(x)[::-1]
    x = np.array(x)[argx]
    y = np.array(y)[argx]
    plt.title("Channel islands class distribution of subset: " + setname + ". Total count: n=" + str(len(dataset)))
    sns.set_context('paper')
    sns.set_color_codes('muted')
    p = sns.barplot(x = x, y = y, color='b')
    show_values(p, "h", space=0)
    plt.show() 

# %%
for img in meta_anno:
    location_counter[img.get('location')] = location_counter.get(img.get('location'), 0) + 1
location_counter = {k: v for k, v in sorted(location_counter.items(), key=lambda item: item[1], reverse=True)}

#%%
# count number of each class  in test20: 
test20_classes = {}
for img in test20:
    test20_classes[img.get('category_id')] = test20_classes.get(img.get('category_id'), 0) + 1
print(test20_classes)

# now in val20:
val20_classes = {}
for img in val20:
    val20_classes[img.get('category_id')] = val20_classes.get(img.get('category_id'), 0) + 1
print(val20_classes)

# import pandas as pd
# locs = pd.DataFrame.from_dict(location_counter, orient='index')
# # %%
# # save locs dataframe to textfile:
# mycontent = locs.to_latex()
# with open('../thesis_files/' + 'locs.tex', 'w') as f:
#     f.write(mycontent)

# %%
