import json
import os
import argparse
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import EncoderRNN, DecoderRNN, S2VTAttModel, S2VTModel
# from dataloader import VideoDataset
from msrvtt_dataset import VideoDataset
import misc.utils as utils
from misc.cocoeval import suppress_stdout_stderr, COCOScorer

# from pandas.io.json import json_normalize
from tqdm import tqdm

"""
def convert_data_to_coco_scorer_format(data_frame):
    gts = {}
    for row in zip(data_frame["caption"], data_frame["video_id"]):
        if row[1] in gts:
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
        else:
            gts[row[1]] = []
            gts[row[1]].append(
                {'image_id': row[1], 'cap_id': len(gts[row[1]]), 'caption': row[0]})
    
    # json.dump(gts, open('./gts0.json', 'w'))
    return gts
# """

# """
def convert_data_to_coco_scorer_format(json_data):
    gts = {}
    for _data in json_data:
        _video_id = _data['video_id']
        _caption = _data['caption']
        if _video_id in gts:
            gts[_video_id].append(
                {
                    'image_id': _video_id, 
                    'cap_id': len(gts[_video_id]), 
                    'caption': _caption
                }
            )
        else:
            gts[_video_id] = []
            gts[_video_id].append(
                {
                    'image_id': _video_id, 
                    'cap_id': len(gts[_video_id]), 
                    'caption': _caption
                }
            )
            
    # json.dump(gts, open('./gts1.json', 'w'))
    return gts
# """

def test(model, crit, test_dataset, vocab, opt):
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=opt["batch_size"], shuffle=True)
    
    # 得分评估器
    scorer = COCOScorer()
    # 作为gt，用于性能评估
    """
    # 将半结构化json数据转换为平面表
    gt_dataframe = json_normalize(
        json.load(open(opt["input_json"]))['sentences'])
    # 将数据转换为coco评估所需的gt格式
    gts = convert_data_to_coco_scorer_format(gt_dataframe)
    """
    # """
    # 将数据转换为coco评估所需的gt格式
    gts = convert_data_to_coco_scorer_format(
        json.load(open(opt["input_json"]))['sentences']
    )
    # """
    results = []
    samples = {}
    for data in tqdm(test_loader):
        # forward the model to get loss
        # 读取数据
        fc_feats = data['fc_feats'].cuda()
        labels = data['labels'].cuda()
        masks = data['masks'].cuda()
        video_ids = data['video_ids']
        
        # forward the model to also get generated samples for each image
        with torch.no_grad():
            seq_probs, seq_preds = model(
                fc_feats, mode='inference', opt=opt)

        # seq_preds: [B, seq_len]
        # 将单词index序列解码为单词
        sents = utils.decode_sequence(vocab, seq_preds)

        # 存储captions，作为pred，用于性能评估
        for k, sent in enumerate(sents):
            video_id = video_ids[k]
            samples[video_id] = [{'image_id': video_id, 'caption': sent}]
            
        json.dump(samples, open('./preds.json', 'w'))

    # 性能评估
    with suppress_stdout_stderr():
        valid_score = scorer.score(gts, samples, samples.keys())
    results.append(valid_score)
    print(valid_score)

    if not os.path.exists(opt["results_path"]):
        os.makedirs(opt["results_path"])

    # 存储scores结果
    with open(os.path.join(opt["results_path"], "scores.txt"), 'a') as scores_table:
        scores_table.write(json.dumps(results[0]) + "\n")
        
    # 存储模型为测试集视频生成的描述 + 评估得分
    with open(os.path.join(opt["results_path"],
                           opt["model"].split("/")[-1].split('.')[0] + ".json"), 'w') as prediction_results:
        json.dump({"predictions": samples, "scores": valid_score},
                  prediction_results)


def main(opt):
    # 构造数据集
    test_dataset = VideoDataset(opt, "test")
    opt["vocab_size"] = test_dataset.get_vocab_size()
    opt["seq_length"] = test_dataset.max_len
    
    # 构造模型
    if opt["model"] == 'S2VTModel':
        model = S2VTModel(opt["vocab_size"], opt["max_len"], 
                          opt["dim_hidden"], opt["dim_word"],
                          rnn_dropout_p=opt["rnn_dropout_p"]).cuda()
    elif opt["model"] == "S2VTAttModel":
        encoder = EncoderRNN(opt["dim_vid"], opt["dim_hidden"], 
                             bidirectional=opt["bidirectional"],
                             input_dropout_p=opt["input_dropout_p"], 
                             rnn_dropout_p=opt["rnn_dropout_p"])
        decoder = DecoderRNN(opt["vocab_size"], opt["max_len"], 
                             opt["dim_hidden"], opt["dim_word"],
                             input_dropout_p=opt["input_dropout_p"],
                             rnn_dropout_p=opt["rnn_dropout_p"], 
                             bidirectional=opt["bidirectional"])
        model = S2VTAttModel(encoder, decoder).cuda()
    # 导入模型参数
    # Setup the model
    model.load_state_dict(torch.load(opt["saved_model"]))
    # 损失计算
    crit = utils.LanguageModelCriterion()
    # 模型测试
    test(model, crit, test_dataset, test_dataset.get_vocab(), opt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', 
                        default='data/test_videodatainfo.json', 
                        type=str, 
                        help='msr_vtt test set json')
    parser.add_argument('--recover_opt', type=str, required=True,
                        help='recover train opts from saved opt_json')
    parser.add_argument('--saved_model', type=str, default='',
                        help='path to saved model to evaluate')

    parser.add_argument('--dump_json', type=int, default=1,
                        help='Dump json with predictions into vis folder? (1=yes,0=no)')
    parser.add_argument('--results_path', type=str, default='results/')
    parser.add_argument('--dump_path', type=int, default=0,
                        help='Write image paths along with predictions into vis json? (1=yes,0=no)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--sample_max', type=int, default=1,
                        help='0/1. whether sample max probs  to get next word in inference stage')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--beam_size', type=int, default=1,
                        help='used when sample_max = 1. Usually 2 or 3 works well.')

    args = parser.parse_args()
    args = vars((args))
    opt = json.load(open(args["recover_opt"]))
    for k, v in args.items():
        opt[k] = v
    main(opt)
