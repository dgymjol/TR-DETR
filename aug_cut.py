from utils.basic_utils import save_jsonl, load_jsonl
from copy import deepcopy
import random
import numpy as np

def crop_clip_index(start_index, end_index, non_idx=False, num_crop=1):
    candidates = list(range(start_index + 2, end_index, 2))
    if non_idx:
        candidates.append(-1) # not crop
    if num_crop > 1:
        return sorted(random.sample(candidates, num_crop))
    else: 
        return random.sample(candidates, num_crop)
    
def find_ones_groups(arr):
    groups = []
    start_idx = None

    l = len(arr)
    for i in range(l):
        if arr[i] == 1 and start_idx is None:
            # 1이 처음 시작되는 인덱스 기록
            start_idx = i
        elif arr[i] == 0 and start_idx is not None:
            # 1의 그룹이 끝나는 인덱스 기록
            groups.append([start_idx * 2, i * 2])
            start_idx = None

    # 마지막 그룹이 배열 끝까지 이어지는 경우 처리
    if start_idx is not None:
        groups.append([start_idx * 2, len(arr) * 2])

    return groups


def find_zeros_groups(arr):
    groups = []
    start_idx = None

    l = len(arr)
    for i in range(l):
        if arr[i] == 0 and start_idx is None:
            # 0이 처음 시작되는 인덱스 기록
            start_idx = i
        elif arr[i] == 1 and start_idx is not None:
            # 0의 그룹이 끝나는 인덱스 기록
            groups.append([start_idx * 2, i * 2])
            start_idx = None

    # 마지막 그룹이 배열 끝까지 이어지는 경우 처리
    if start_idx is not None:
        groups.append([start_idx * 2, len(arr) * 2])

    return groups





max_v_l = 75
seed = 0
random.seed(seed)
np.random.seed(seed)

org_datalist = load_jsonl('data/highlight_train_release.jsonl')
new_datalist = []


for data in org_datalist:

    new_datalist.append(deepcopy(data))
    
    ctx_l = min(data['duration'] // 2, max_v_l)

    ###############################################
    # moment와 non-moment 구하기
    ###############################################

    all_clips = np.zeros(ctx_l)
    all_clips[data['relevant_clip_ids']] = 1

    moments = find_ones_groups(all_clips)
    assert moments == data['relevant_windows']

    non_moments = find_zeros_groups(all_clips)
    # 만약 non-moment가 없다면 이 data는 pass
    if not non_moments:
        continue




    ###############################################
    # 20 이상인 moment 구하기
    ###############################################

    max_moment_length = 0
    max_moment_idx = -1
    for i, (s, e) in enumerate(moments):
        l = e - s
        if max_moment_length < l:
            max_moment_length = l
            max_moment_idx = i

    # 20 이상인 것이 없다면 쪼갤 수 없으므로 이 data는 pass
    if max_moment_length < 20:
        continue

    ###############################################
    # 20 이상인 moment 쪼개고 moment segment 만들기
    ###############################################

    s, e = moments[max_moment_idx]
    num_crop = max_moment_length // 10 - 1
    moment_crop_idxs = crop_clip_index(s, e, num_crop=num_crop)

    moment_segments = []

    ss_idx = 0
    for i, (s, e) in enumerate(moments):
        if i == max_moment_idx:
            moment_crop_idxs.append(e)
            ss = s
            for ee in moment_crop_idxs:
                moment = dict()

                moment['clip_id'] = [(ss // 2 if ss != 0 else 0), ee // 2]
                moment['len'] = (ee - ss) // 2

                ss_nxt_idx = ss_idx + moment['len']
                moment['saliency_scores'] = data['saliency_scores'][ss_idx : ss_nxt_idx]
                ss_idx = ss_nxt_idx

                moment_segments.append(moment)
                ss = ee
        else:
            moment = dict()

            moment['clip_id'] = [(s // 2 if s != 0 else 0), e // 2]
            moment['len'] = (e - s) // 2

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
        l = e - s
        if l >= 20:
            num_crop = min(l // 10 - 1, need_crop_count)
            non_moment_crop_idxs.append(crop_clip_index(s, e, num_crop=num_crop))
            non_moment_idxs.append(i)

            need_crop_count -= num_crop
        
        if need_crop_count <= 0:
            break
    # crop한 moment 사이에 끼워넣을 non-moment가 충분하지 않다면, 이 data는 pass
    if need_crop_count > 0 :
        continue


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

                non_moment['clip_id'] = [(ss // 2 if ss != 0 else 0), ee // 2]
                non_moment['len'] = (ee - ss) // 2

                non_moment_segments.append(non_moment)
                ss = ee
        else:
            non_moment = dict()

            non_moment['clip_id'] = [(s // 2 if s != 0 else 0), e // 2]
            non_moment['len'] = (e - s) // 2

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
    new_data['saliency_scores'] = []
    for i in range(len(moment_segments)):

        # non-moment segment
        non_moment_segment = non_moment_segments[i]
        cur_clip_id += non_moment_segment['len']
        new_data['org_clip_ids_order'].append(non_moment_segment['clip_id'])

        # moment segment
        moment_segment = moment_segments[i]
        nxt_clip_id = cur_clip_id + moment_segment['len']
        new_clips[cur_clip_id:nxt_clip_id] = 1
        cur_clip_id = nxt_clip_id
        new_data['org_clip_ids_order'].append(moment_segment['clip_id'])
        new_data['saliency_scores'] += moment_segment['saliency_scores']

    non_moment_segment = non_moment_segments[-1]
    new_data['org_clip_ids_order'].append(non_moment_segment['clip_id'])

    # new_data['relevant_clip_ids']
    new_data['relevant_clip_ids'] = np.where(new_clips == 1)[0].tolist()
    # new_data['relevant_windows']
    new_data['relevant_windows'] = find_ones_groups(new_clips)

    assert len(data['saliency_scores']) == len(new_data['saliency_scores'])
    assert len(new_data['saliency_scores']) == len(new_data['relevant_clip_ids'])

    new_datalist.append(new_data)


print(f"Oracle Crop : {len(org_datalist)} -> {len(new_datalist)}")
save_jsonl(new_datalist, f'data/highlight_train_aug_release_seed_{seed}.jsonl')
