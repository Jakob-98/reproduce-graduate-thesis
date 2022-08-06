# This file prepares the dicts for the ENA dataset.
#%%
import json
import random
import numpy as np
from operator import itemgetter
from itertools import groupby
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# seeds for reproducibility
random.seed(42)
np.random.seed(42)

class config:
    save_pickles = False  
    metadata_path = 'C:\Projects\wild\data\ENA24\ena24_public.json'
    pickle_path = 'C:\Projects\seq2bbox\data\pickle\ENA\\'

with open(config.metadata_path) as f:
    d = json.load(f)

# changing the labels to remove human labels but keep [1,..., n] order
d['categories'][-1]['id'] = 8
for anno in d['annotations']:
    if anno.get('category_id') == 22:
        anno['category_id'] = 8


# merging the annotations and image metadata
for i in range(len(d['images'])):
    d['images'][i]['image_id'] = d['images'][i].pop('id')

my_id = itemgetter('image_id')
meta_anno = []

for k, v in groupby(sorted((d['annotations'] + d['images']), key=my_id), key=my_id):
    meta_anno.append({key:val for d in v for key, val in d.items()})



# # removing missing images|| UPDATE: This does not work, and messes with label=0
# missing_ids = set()
# for idx, image in enumerate(meta_anno):
#     if not image.get('category_id'):
#         print(image.get('category_id'))
#         missing_ids.add(image['image_id'])
# print('number of missing ids: {}'.format(len(missing_ids)))
# meta_anno = [img for img in meta_anno if img.get('image_id') not in missing_ids]

#%%
# count the number of images in each category:
counts = {}
for anno in meta_anno:
    if anno.get('category_id') in counts:
        counts[anno.get('category_id')] += 1
    else:
        counts[anno.get('category_id')] = 1
print(counts)
#%%

# Generating subsets... 
full_length = len(meta_anno)
labels = [i.get('category_id') for i in meta_anno]
train, test = train_test_split(meta_anno, test_size=(full_length//5), random_state=42, stratify=labels)
labels = [i.get('category_id') for i in train]
train, validate = train_test_split(train, test_size=(full_length//10), random_state=42, stratify=labels)
labels = [i.get('category_id') for i in train]
full_train = len(train)
trainsubsets = [train]
for i in (0.5, 0.8, 0.9, 0.95): 
    subset, throwaway = train_test_split(train, test_size=i, random_state=42, stratify=labels)
    trainsubsets.append(subset)

#saving the subsets to pickle
if config.save_pickles:
    for dataset, setname in zip(trainsubsets + [validate, test], ('train100', 'train50', 'train20', 'train10', 'train5', "val", "test")):
        with open(config.pickle_path + setname + '.pk', 'wb') as f:
            pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


# EDA....
# https://www.statology.org/seaborn-barplot-show-values/
# plotting...

# %%
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


# for dataset, setname in zip(meta_anno + trainsubsets + [validate, test], ('full_set', 'train100', 'train50', 'train20', 'train10', 'train5', "val", "test")):
for dataset, setname in zip(trainsubsets + [validate, test, meta_anno], ('train100', 'train50', 'train20', 'train10', 'train5', "val", "test", "full_set")):
    location_counter = {}
    sequence_counter = {}
    species_counter = {}

    print(type(dataset))

    for img in dataset:
        species_counter[img.get('category_id')] = species_counter.get(img.get('category_id'), 0) + 1

    species_lookup = {i.get('id') : i.get('name') for i in d['categories']}
    species_named_counter = {species_lookup.get(k): v for k, v in species_counter.items()}

    plt.figure(num=None, figsize=(10,8), dpi=80, facecolor='w', edgecolor='r')
    y = list(species_named_counter.keys())
    x = list(species_named_counter.values())
    argx = np.argsort(x)[::-1]
    x = np.array(x)[argx]
    y = np.array(y)[argx]
    plt.title("ENA24 class distribution of subset: " + setname + ". Total count: n=" + str(len(dataset)))
    sns.set_context('paper')
    sns.set_color_codes('muted')
    p = sns.barplot(x = x, y = y, color='b')
    show_values(p, "h", space=0)
    plt.show() 


# # %%
# # print val set label counter: 
# val_counter = {}
# for img in validate:
#     val_counter[img.get('category_id')] = val_counter.get(img.get('category_id'), 0) + 1

# # sort counter and print: 
# val_counter = sorted(val_counter.items(), key=itemgetter(0), reverse=False)
# print(val_counter)
# # %%
# val_counter == [(1, 30), (2, 31), (3, 21), (4, 29), (5, 34), (6, 72), (7, 28), (8, 89), (9, 14), (10, 30), (11, 41), (12, 34), (13, 29), (14, 42), (15, 6), (16, 70), (17, 92), (18, 52), (19, 48), (20, 33), (21, 33)]
# # %%
# # count number of birds (category 0) in d['annotations']:
# count = 0
# for img in d['annotations']:
#     if img.get('category_id') == 9:
#         count += 1
# print(count)
# # %%

# %%
