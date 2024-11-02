from utils.basic_utils import save_jsonl, load_jsonl
from copy import deepcopy
import random
import numpy as np
from utils.length_aug import *
import sys

def temp_mix(data, moments, non_moments, thres_crop, ctx_l, clip_len):
    '''
    In a video, crop one longer moment and mix non-moment and moments
    
    '''

    ###############################################
    # 20 이상인 moment 구하기
    ###############################################

    max_moment_length = 0
    max_moment_idx = -1
    ms, me = -1, -1
    for i, (s, e) in enumerate(moments):
        rs = int(s // clip_len) if s != 0 else 0
        re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
        rs, re = rs * clip_len, re * clip_len

        l = re - rs
        if max_moment_length < l:
            max_moment_length = l
            max_moment_idx = i
            ms, me = rs, re

    # 20 이상인 것이 없다면 쪼갤 수 없으므로 이 data는 pass
    if max_moment_length < thres_crop * 2:
        return None

    ###############################################
    # 20 이상인 moment 쪼개고 moment segment 만들기
    ###############################################

    num_crop = max_moment_length // thres_crop - 1
    moment_crop_idxs = crop_clip_index(ms, me, num_crop=num_crop, clip_len=clip_len)

    moment_segments = []

    ss_idx = 0
    for i, (s, e) in enumerate(moments):
        if i == max_moment_idx:
            moment_crop_idxs.append(e)
            ss = s
            for ee in moment_crop_idxs:
                moment = dict()
                
                rss = int(ss // clip_len) if ss != 0 else 0
                ree = int(ee // clip_len) if ee % clip_len == 0 else int(ee // clip_len) + 1
                if clip_len < 1 and s != ss: # vgg
                    rss += 1
                moment['clip_id'] = [rss, ree]
                moment['seg_sec'] = [ss - rss * clip_len, ree * clip_len - ee]
                moment['len'] = (ree - rss)

                if 'saliency_scores' in data:
                    ss_nxt_idx = ss_idx + moment['len']
                    moment['saliency_scores'] = data['saliency_scores'][ss_idx : ss_nxt_idx]
                    ss_idx = ss_nxt_idx

                moment_segments.append(moment)
                ss = ee
        else:
            moment = dict()

            rs = int(s // clip_len) if s != 0 else 0
            re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
            moment['clip_id'] = [rs, re]
            moment['seg_sec'] = [s - rs * clip_len, re * clip_len - e]
            moment['len'] = (re - rs)

            if 'saliency_scores' in data:
                ss_nxt_idx = ss_idx + moment['len']
                moment['saliency_scores'] = data['saliency_scores'][ss_idx : ss_nxt_idx]
                ss_idx = ss_nxt_idx

            moment_segments.append(moment)


    ###############################################
    # 20 이상인 non_moments들을 필요한 만큼 쪼개기
    ###############################################

    need_crop_count = len(moment_segments) + 1 - len(non_moments)

    non_moment_crop_idxs = []
    non_moment_idxs = []

    for i, (s, e) in enumerate(non_moments):

        if s == 0:
            rs = 0
        else:
            rs = int(s // clip_len) if s % clip_len == 0 else int(s // clip_len) + 1
        re = int(e // clip_len)
        rs, re = rs * clip_len, re * clip_len

        l = re - rs
        if l >= thres_crop * 2:
            num_crop = min(l // thres_crop - 1, need_crop_count)
            non_moment_crop_idxs.append(crop_clip_index(rs, re, num_crop=num_crop, clip_len=clip_len))
            non_moment_idxs.append(i)

            need_crop_count -= num_crop
        
        if need_crop_count <= 0:
            break
    # crop한 moment 사이에 끼워넣을 non-moment가 충분하지 않다면, 이 data는 pass
    if need_crop_count > 0 :
        return None


    ###############################################
    # non_moment_segments 만들기
    ###############################################

    non_moment_segments = []

    for i, (s, e) in enumerate(non_moments):
        if i in non_moment_idxs:
            _non_moment_crop_idxs = non_moment_crop_idxs[non_moment_idxs.index(i)]
            _non_moment_crop_idxs.append(e)
            
            ss = s
            for ee in _non_moment_crop_idxs:
                non_moment = dict()
                
                if ss == 0:
                    rss = 0
                else:
                    rss = int(ss // clip_len) if ss % clip_len == 0 else int(ss // clip_len) + 1
                ree = int(ee // clip_len)
                if clip_len < 1 and s != ss: # vgg
                    rss -= 1

                non_moment['clip_id'] = [rss, ree]
                non_moment['len'] = (ree - rss)

                non_moment_segments.append(non_moment)
                ss = ee
        else:
            non_moment = dict()

            if s == 0:
                rs = 0
            else:
                rs = int(s // clip_len) if s % clip_len == 0 else int(s // clip_len) + 1
            re = int(e // clip_len)

            non_moment['clip_id'] = [rs, re]
            non_moment['len'] = (re - rs)

            non_moment_segments.append(non_moment)



    ###############################################
    # moment와 non-moment 섞기
    ###############################################

    random.shuffle(non_moment_segments)
    random.shuffle(moment_segments)

    ###############################################
    # 새로운 data 만들기
    ###############################################

    new_data = dict()
    new_data['qid'] = data['qid']
    new_data['query'] = data['query']
    new_data['duration'] = data['duration']
    new_data['vid'] = data['vid']

    new_clips = np.zeros(ctx_l)

    # new_data['saliency_scores'] ok
    # new_data['org_clip_ids_order'] ok
    cur_clip_id = 0
    new_data['org_clip_ids_order'] = []
    if 'saliency_scores' in data:
        new_data['saliency_scores'] = []
    seg_secs = []
    for i in range(len(moment_segments)):

        # non-moment segment
        non_moment_segment = non_moment_segments[i]
        cur_clip_id += non_moment_segment['len']
        new_data['org_clip_ids_order'].append((data['vid'], non_moment_segment['clip_id']))

        # moment segment
        moment_segment = moment_segments[i]
        nxt_clip_id = cur_clip_id + moment_segment['len']
        new_clips[cur_clip_id:nxt_clip_id] = 1
        cur_clip_id = nxt_clip_id
        new_data['org_clip_ids_order'].append((data['vid'], moment_segment['clip_id']))
        if 'saliency_scores' in data:
            new_data['saliency_scores'] += moment_segment['saliency_scores']
        seg_secs.append(moment_segment['seg_sec'])

    non_moment_segment = non_moment_segments[-1]
    new_data['org_clip_ids_order'].append((data['vid'], non_moment_segment['clip_id']))

    if 'relevant_clip_ids' in data:
        new_data['relevant_clip_ids'] = np.where(new_clips == 1)[0].tolist()
        new_data['relevant_windows'] = find_ones_groups(new_clips, clip_len)
    else:
        new_data['relevant_windows'] = []
        for sup_m, sub_m in zip(find_ones_groups(new_clips, clip_len), seg_secs):
            sups, supe = sup_m
            subs, sube = sub_m
            new_data['relevant_windows'].append([sups + subs, supe - sube])


    ### Test ####
    if 'saliency_scores' in data:
        assert len(data['saliency_scores']) == len(new_data['saliency_scores'])
        assert len(new_data['saliency_scores']) == len(new_data['relevant_clip_ids'])

    clips_for_check = np.zeros(ctx_l)
    for _, (s, e) in new_data['org_clip_ids_order']:
        clips_for_check[s:e] += 1
    assert np.all(clips_for_check <= 1)
    assert np.all(clips_for_check[:-1] > 0)

    return new_data




def feat_mix(data, moments, non_moments, ctx_l, clip_len, db_range, moment_db):
    '''
    to replace Non GT to other videos' GT
    
    '''

    ###############################################
    # non_moment_segments replacement
    ###############################################

    non_moment_segments = []

    for i, (s, e) in enumerate(non_moments):
        non_moment = dict()

        need_len = (e- s)

        find = False
        db_range_idx = -1
        for db_range_ in db_range:
            if need_len > db_range_:
                find = True
                break
            db_range_idx += 1

        if not find or db_range_idx == -1:
            print(need_len)
            return None
        
        another_moment = random.choice(moment_db[db_range_idx])
        non_moment['vid'] = another_moment[0]


        ass, aee = another_moment[1]
        if aee - ass < need_len:
            assert False
        else:
            aee = ass + need_len

        if ass == 0:
            rss = 0
        else:
            rss = int(ass // clip_len) if ass % clip_len == 0 else int(ass // clip_len) + 1

        ree = int(aee // clip_len)


        non_moment['clip_id'] = [rss, ree]
        non_moment['len'] = (ree - rss)

        non_moment_segments.append(non_moment)


    ###############################################
    # 새로운 data 만들기
    ###############################################

    new_data = dict()
    new_data['qid'] = data['qid']
    new_data['query'] = data['query']
    new_data['duration'] = data['duration']
    new_data['vid'] = data['vid']
    new_data['relevant_windows'] = data['relevant_windows']

    if 'relevant_clip_ids' in data:
        new_data['relevant_clip_ids'] = data['relevant_clip_ids']

    if 'saliency_scores' in data:
        new_data['saliency_scores'] = data['saliency_scores']


    new_data['org_clip_ids_order'] = []

    non_moments_idx = 0
    if non_moments[0][0] == 0:
        non_moment_segment = non_moment_segments[0]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))
        non_moments_idx += 1

    for i in range(len(moments) - 1):

        # moment segment
        s, e = moments[i]
        rs = int(s // clip_len) if s != 0 else 0
        re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
        new_data['org_clip_ids_order'].append((data['vid'], [rs, re]))

        # non-moment segment
        non_moment_segment = non_moment_segments[non_moments_idx]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))
        non_moments_idx += 1

    # moment segment
    s, e = moments[-1]
    rs = int(s // clip_len) if s != 0 else 0
    re = int(e // clip_len) if e % clip_len == 0 else int(e // clip_len) + 1
    new_data['org_clip_ids_order'].append((data['vid'], [rs, re]))

    if non_moments_idx < len(non_moment_segments):
        non_moment_segment = non_moment_segments[-1]
        new_data['org_clip_ids_order'].append((non_moment_segment['vid'], non_moment_segment['clip_id']))

    assert len(new_data['org_clip_ids_order']) == len(moments) + len(non_moments)
    return new_data



dset_name = sys.argv[1]
seed = int(sys.argv[2])
thres_crop = int(sys.argv[3])


print(f" ========== {dset_name} augmentation =============")
print(f" seed : {seed}")
print(f" thres_crop : {thres_crop}")

savefilename = f"data/{dset_name}"
savefilename += f"_temp_and_feat_mix_{thres_crop}"
savefilename += f"_seed_{seed}.jsonl"

random.seed(seed)
np.random.seed(seed)



if dset_name == 'hl':
    datalist = load_jsonl('data/highlight_train_release.jsonl')
    clip_len = 2
    db_range = [150, 130, 110, 90, 70, 50, 30, 10, 0] # moment class borderline

    print(f" dset : QVHighlights")

elif dset_name == 'tacos':
    # tacos-train min: 0.4761904761904816 max: 751.4285714285714
    datalist = load_jsonl('data/tacos/train.jsonl')
    clip_len = 2
    db_range = [700, 500, 300, 200, 150, 130, 110, 90, 70, 50, 30, 10, 0] # moment class borderline

    print(f" dset : TACoS")

elif 'charades' in dset_name :
    # cha-train min: 1.6799999999999997 max: 80.80000000000001 
    datalist = load_jsonl('data/charades/charades_sta_train_tvr_format.jsonl')
    db_range = [80, 50, 30, 20, 10, 0] # moment class borderline

    if 'vgg' in dset_name:
        clip_len = 0.166666
        print(f" dset : Charades - VGG")

    else:
        clip_len = 1
        print(f" dset : Charades")

elif 'nlq' in dset_name :
    datalist = load_jsonl('data/nlq/train.jsonl')
    clip_len = 2
    print(f" dset : NLQ")

else:
    assert False

print(f" ============================================")


# another video moment database
moment_db = [[] for i in range(len(db_range))]

for data in datalist:
    moments = data['relevant_windows']
    for start, end in moments:
        for i, db_range_value in enumerate(db_range):

            if (end-start) >= db_range_value:
                moment_db[i].append((data['vid'], [start, end]))
                break
    
print(f"Moment Database")
for db_range_, moment_db_ in zip(db_range, moment_db):
    print(f"moment db (>= {db_range_}) : {len(moment_db_)}")


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
    new_crop_data = temp_mix(data, moments=moments, non_moments=non_moments, thres_crop=thres_crop, ctx_l=ctx_l, clip_len=clip_len)
    if new_crop_data:
        new_datalist.append(new_crop_data)

    new_replace_data = feat_mix(data, moments=moments, non_moments=non_moments, ctx_l=ctx_l, clip_len=clip_len, db_range=db_range, moment_db=moment_db)
    if new_replace_data:
        new_datalist.append(new_replace_data)

print(f"Length Augmentation : {len(datalist)} -> {len(new_datalist)}")
print(f"Saved File : {savefilename}")

save_jsonl(new_datalist, savefilename)
