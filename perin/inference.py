#!/usr/bin/env python3
# coding=utf-8

import argparse
import torch
import os

from model.model import Model
from data.dataset import Dataset
from utility.initialize import initialize
from config.params import Params
from utility.predict import predict



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_directory", type=str, default="../dataset")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir

    checkpoint = torch.load(f"{checkpoint_dir}/best_checkpoint.h5", map_location=torch.device('cpu'))
    args = Params().load_state_dict(checkpoint["args"]).init_data_paths()
    args.log_wandb = False



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Dataset(args, verbose=False)

    model = Model(dataset, args).to(device)
    model.load_state_dict(checkpoint["model"])

    os.makedirs(f"{checkpoint_dir}/inference", exist_ok=True)

    print("inference of test data", flush=True)

    predict(model, dataset.test, args.test_data, args.raw_testing_data, args, None, f"{checkpoint_dir}/inference", device, mode="test")
