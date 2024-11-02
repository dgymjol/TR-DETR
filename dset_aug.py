from utils.basic_utils import save_jsonl, load_jsonl
from copy import deepcopy
import random
import numpy as np
from utils.length_aug import *
import sys

dset_name = sys.argv[1]
seed = int(sys.argv[2])
crop_ = eval(sys.argv[3])
assert isinstance(crop_, bool)
merge_ = eval(sys.argv[4])
assert isinstance(crop_, bool)
thres_crop = int(sys.argv[5])
thres_merge = int(sys.argv[6])

print(f" ========== {dset_name} augmentation =============")
print(f" seed : {seed}")
print(f" crop : {crop_}")
print(f" merge : {merge_}")
print(f" thres_crop : {thres_crop}")
print(f" thres_merge : {thres_merge}")

savefilename = f"data/{dset_name}"
if crop_:
    savefilename += f"_crop_{thres_crop}"
if merge_:
    savefilename += f"_merge_{thres_merge}"
savefilename += f"_seed_{seed}.jsonl"

random.seed(seed)
np.random.seed(seed)

print(f" ============================================")


if dset_name == 'hl':
    datalist = load_jsonl('data/highlight_train_release.jsonl')
    clip_len = 2

elif dset_name == 'tacos':
    datalist = load_jsonl('data/tacos/train.jsonl')
    clip_len = 2

elif dset_name in 'charades':
    datalist = load_jsonl('data/charades/charades_sta_train_tvr_format.jsonl')
    if 'vgg' in dset_name:
        clip_len = 1.6666
    else:
        clip_len = 1
else:
    assert False


new_datalist = []

for data in datalist:

    new_datalist.append(deepcopy(data))

    ctx_l = int(data['duration'] // clip_len) if data['duration'] % clip_len == 0 else int(data['duration'] // clip_len) + 1

    ###############################################
    # moment와 non-moment 구하기
    ###############################################

    if 'relevant_clip_ids' in data: # QVHighlights

        all_clips = np.zeros(ctx_l)
        all_clips[data['relevant_clip_ids']] = 1

        moments = find_ones_groups(all_clips, clip_len)
        assert moments == data['relevant_windows']

        non_moments = find_zeros_groups(all_clips, clip_len)

    else: # Charades, TACoS (single moment)
        moments = data['relevant_windows']
        non_moments = []
        if moments[0][0] != 0:
            non_moments.append([0, moments[0][0]])
        if moments[0][1] != data['duration']:
            non_moments.append([moments[0][1], data['duration']])    

    # 만약 non-moment가 없다면 이 data는 pass
    if not non_moments:
        continue 
    
    # crop augmentation
    if crop_:
        new_crop_data = crop(data, moments=moments, non_moments=non_moments, thres_crop=thres_crop, ctx_l=ctx_l, clip_len=clip_len)
        if new_crop_data:
            new_datalist.append(new_crop_data)

    # merge augmentation for multi-moments dataset
    if merge_:
        if dset_name == 'hl': 
            new_merge_data = merge_multi_moments(data, moments=moments, non_moments=non_moments, thres_merge=thres_merge, ctx_l=ctx_l, clip_len=clip_len)

            if new_merge_data:
                new_datalist.append(new_merge_data)
        else:
            s, e = data['relevant_windows'][0]
            rs = int(s // clip_len) if s % clip_len == 0 else int(s // clip_len) + 1
            re = int(e // clip_len)
            rs, re = rs * clip_len, re * clip_len

            moments = [[rs, re]]
            non_moments = []
            if rs - clip_len > 0:
                non_moments.append([0, rs - clip_len])
            if re + clip_len < ctx_l * clip_len:
                non_moments.append([re + clip_len, ctx_l * clip_len ])    

            new_merge_data = merge_single_moment(data, moments=moments, non_moments=non_moments, thres_merge=thres_merge, ctx_l=ctx_l, clip_len=clip_len)
            if new_merge_data:
                new_datalist.append(new_merge_data)

print(f"Length Augmentation : {len(datalist)} -> {len(new_datalist)}")
print(f"Saved File : {savefilename}")

save_jsonl(new_datalist, savefilename)
