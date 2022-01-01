import re
import json
import argparse
import numpy as np


def build_vocab(vids, params):
    # 词汇统计阈值，只统计出现次数超过该值的单词
    count_thr = params['word_count_threshold']
    # 统计单词出现次数
    counts = {}
    # vids数据组织形式：{video_id : {'captions': []}}
    for vid, caps in vids.items():
        for cap in caps['captions']:
            # 正则表达式，将符号[.!,;?]替换为空格，然后按单词划分
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            for w in ws:
                counts[w] = counts.get(w, 0) + 1

    total_words = sum(counts.values())
    bad_words = [w for w, n in counts.items() if n < count_thr]
    vocab = [w for w, n in counts.items() if n >= count_thr]
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' %
          (len(bad_words), len(counts), len(bad_words) * 100.0 / len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' %
          (bad_count, total_words, bad_count * 100.0 / total_words))
    # lets now produce the final annotations
    if bad_count > 0:
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('<UNK>')
        
    # 在vids数据中，插入final_captions项，即每个视频描述的划分单词
    # 包含<sos> <eos> <UNK>
    for vid, caps in vids.items():
        caps = caps['captions']
        vids[vid]['final_captions'] = []
        for cap in caps:
            ws = re.sub(r'[.!,;?]', ' ', cap).split()
            caption = [
                '<sos>'] + [w if counts.get(w, 0) > count_thr else '<UNK>' for w in ws] + ['<eos>']
            vids[vid]['final_captions'].append(caption)
    return vocab


def main(params):
    if len(params['input_json']) == 1:
        videos_json = json.load(open(params['input_json'][0], 'r'))
    else:
        videos_json = json.load(open(params['input_json'][0], 'r'))
        _json = json.load(open(params['input_json'][1], 'r'))
        videos_json['videos'] += _json['videos']
        videos_json['sentences'] += _json['sentences']
    
    # 处理视频的caption
    # 1、收集每个视频所对应的caption list，存储到video_caption中
    videos_sentences = videos_json['sentences']
    video_caption = {}
    for _data in videos_sentences:
        if _data['video_id'] not in video_caption.keys():
            video_caption[_data['video_id']] = {'captions': []}
        video_caption[_data['video_id']]['captions'].append(_data['caption'])

    # 2、创建词汇表
    vocab = build_vocab(video_caption, params)
    itow = {i + 2: w for i, w in enumerate(vocab)}
    wtoi = {w: i + 2 for i, w in enumerate(vocab)}  # inverse table
    wtoi['<eos>'] = 0
    itow[0] = '<eos>'
    wtoi['<sos>'] = 1
    itow[1] = '<sos>'

    # 处理词汇相关信息，存储到out中
    out = {}
    out['ix_to_word'] = itow
    out['word_to_ix'] = wtoi
    # 将数据划分为train / val / test
    out['videos'] = {'train': [], 'val': [], 'test': []}
    videos_infos = videos_json['videos']
    for _data in videos_infos:
        split = 'val' if _data['split'] == 'validate' else _data['split']
        # out['videos'][split].append(int(_data['id']))  # 仅使用MSR-VTT，存储视频的index
        out['videos'][split].append(_data['video_id'])   # 直接存储视频的id
    json.dump(out, open(params['info_json'], 'w'))
    json.dump(video_caption, open(params['caption_json'], 'w'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', 
                        default=['data/train_val_videodatainfo.json', 'data/test_videodatainfo.json'], 
                        nargs='+',
                        type=str, 
                        help='msr_vtt videoinfo json')
    parser.add_argument('--info_json', default='data/info.json',
                        help='info about iw2word and word2ix')
    parser.add_argument('--caption_json', default='data/caption.json', help='caption json file')


    parser.add_argument('--word_count_threshold', default=5, type=int,
                        help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    main(params)
