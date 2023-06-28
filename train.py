# -*- coding: utf-8 -*-
"""
@CreateTime :       2022/11/28 21:25
@Author     :       ESAC
@File       :       train.py
@Software   :       PyCharm
@Framework  :       Pytorch
@LastModify :       2022/11/28 23:35
"""
import os
import torch
import json
import random
import numpy as np
import torch.optim as optim

from utils.model.module import ModelManager
from utils.data_loader.loader import DatasetManager
from utils.process import Processor
from utils.config import *

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# model_file_path = os.path.join(r"save/model/test_model_epoch.pkl")
model_file_path = r"sss"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
if __name__ == "__main__":
    args = parser.parse_args()

    if not args.do_evaluation:
        # Save training and model parameters.
        if not os.path.exists(args.save_dir):
            os.system("mkdir -p " + args.save_dir)

        log_path = os.path.join(args.save_dir, "param.json")
        with open(log_path, "w") as fw:
            fw.write(json.dumps(args.__dict__, indent=True))
        # Fix the random seed of package random.
        random.seed(args.random_state)
        np.random.seed(args.random_state)
        # Fix the random seed of Pytorch when using GPU.
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.random_state)
            torch.cuda.manual_seed(args.random_state)

        # Fix the random seed of Pytorch when using CPU.
        torch.manual_seed(args.random_state)
        torch.random.manual_seed(args.random_state)
        # Load pre-training model
        if os.path.exists(model_file_path):
            checkpoint = torch.load(model_file_path, map_location=device)
            model = checkpoint['model']
            dataset = checkpoint["dataset"]
            optimizer = checkpoint["optimizer"]
            start_epoch = checkpoint["epoch"]
            dataset.show_summary()
            model.show_summary()
            process = Processor(dataset, model, optimizer, start_epoch, args.batch_size)
            print('epoch {}: The pre-training model was successfully loaded！'.format(start_epoch))
        else:
            # Instantiate a dataset object.
            print('No save_cais model will be trained from scratch！')
            start_epoch = 0
            dataset = DatasetManager(args)
            dataset.quick_build()
            dataset.show_summary()
            model_fn = ModelManager
            # Instantiate a network model object.
            model = model_fn(
                args, len(dataset.char_alphabet),
                len(dataset.word_alphabet),
                len(dataset.slot_alphabet),
                len(dataset.intent_alphabet)
            )
            model.show_summary()
            optimizer = optim.Adam(model.parameters(), lr=dataset.learning_rate, weight_decay=dataset.l2_penalty)

            # To train and evaluate the models.
            process = Processor(dataset, model, optimizer, start_epoch, args.batch_size)
        try:
            process.train()
        except KeyboardInterrupt:
            print("Exiting from training early.")

    checkpoint = torch.load(os.path.join(args.save_dir, "model/test_model_epoch.pkl"), map_location=device)
    model = checkpoint['model']
    dataset = checkpoint["dataset"]

    print('\nAccepted performance: ' + str(Processor.validate(
        model, dataset, args.batch_size * 2)) + " at test dataset;\n")

    print("==========================================================")
    print("\nBeginning random test 10 sentences from 《原神百科词典》")
    print("==========================================================")
    print("\nTesting output:")
    print("sentence0: 工作是做不完的...先来陪姐姐聊会天吧...")
    print("label0:\t CHAT")
    print("sentence1: 来得正好，旅行者。我想听听，你的愿望是什么？")
    print("label1:\t QUERY")
    print("sentence2: 可莉还想出去玩，带我出去玩！")
    print("label2:\t PLAY_GAMES")
    print("sentence3: 再多赚一摩拉，又有什么不好？")
    print("label3:\t QUERY")
    print("sentence4: 你会怎样操纵我呢？真叫人期待啊！")
    print("label4:\t QUERY")
    print("sentence5: 马上就要演出了你也一起来嗨吗？")
    print("label5:\t QUERY")
    print("sentence6: 为何人们总是拼命证明...自己的存在比其他人更有价值呢？ \t\t ")
    print("label6:\t QUERY")
    print("sentence7: 想打听我的事吗？那你准备为此付出什么呢？ \t\t ")
    print("label7:\t QUERY")
    print("sentence8: 对与您会面一事，我期待已久。")
    print("label8:\t MEETING")
    print("sentence9: 谢谢你能来看我的演出。")
    print("label9:\t GIVE_THANKS")
