import model
import dataset
import numpy as np
import torch
import cv2
import torch.nn as nn
from dataset.data_set import CUDAPrefetcher, PairedImageDataset
import random
import skimage.measure as sk
from skimage import morphology
import matplotlib.pyplot as plt
import tifffile as tif
from torchvision.utils import save_image
from skimage import io
from PIL import Image
import argparse
from utils import utils
import time
from typing import Any
from torch.utils.data import DataLoader
import yaml
from typing import  Any
# def Totensor(x):
#     x = np.transpose(x,(2,0,1))
#     x = torch.from_numpy(x).float()/255.0
#     return x

def load_dataset(config: Any, device: torch.device) -> CUDAPrefetcher:
    test_A_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_A_IMAGES_DIR"])
    test_B_datasets = PairedImageDataset(config["TEST"]["DATASET"]["PAIRED_TEST_B_IMAGES_DIR"])
    test_A_dataloader = DataLoader(test_A_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    test_B_dataloader = DataLoader(test_B_datasets,
                                 batch_size=config["TEST"]["HYP"]["IMGS_PER_BATCH"],
                                 shuffle=config["TEST"]["HYP"]["SHUFFLE"],
                                 num_workers=config["TEST"]["HYP"]["NUM_WORKERS"],
                                 pin_memory=config["TEST"]["HYP"]["PIN_MEMORY"],
                                 drop_last=False,
                                 persistent_workers=config["TEST"]["HYP"]["PERSISTENT_WORKERS"])
    test_A_prefetcher = CUDAPrefetcher(test_A_dataloader, device)
    test_B_prefetcher = CUDAPrefetcher(test_B_dataloader, device)

    return test_A_prefetcher,test_B_prefetcher

def build_model(opts: Any, device: torch.device):
    attention = model.Generator()
    attention.eval()
    attention.to(torch.device("cuda"))
    gather = model.gather_module(opts, attention)
    gather.load_state_dict(torch.load(opts["TEST"]["Weights"]["path"]))
    g_model = gather.to(device)


    # # compile model
    # if config["MODEL"]["G"]["COMPILED"]:
    #     g_model = torch.compile(g_model)

    return g_model

def test(
        g_model:nn.Module,
        input_img_a:CUDAPrefetcher,
        input_img_b:CUDAPrefetcher,
        config:Any
):
    save_image=False
    batches=len(input_img_a)
    if config["TEST"]["SAVE_IMAGE_DIR"]:
        save_image=True
        utils.make_directory(config["TEST"]["SAVE_IMAGE_DIR"])
    batch_time = utils.AverageMeter("Time", ":6.3f", utils.Summary.NONE)
    progress = utils.ProgressMeter(len(input_img_a),
                             [batch_time],
                             prefix=f"Test: ")
    g_model.eval()
    g_model.to(torch.device("cuda"))
    with torch.no_grad():
        batch_index=0
        # Set the data set iterator pointer to 0 and load the first batch of data
        input_img_a.reset()
        batch_data_a = input_img_a.next()

        input_img_b.reset()
        batch_data_b = input_img_b.next()

        # Record the start time of verifying a batch
        end = time.time()
        while batch_data_a is not None and batch_data_b is not None:
            if batch_index>batches-1:
                break
            pair_a=batch_data_a["lr"]
            pair_b=batch_data_b["lr"]
            x_out, y_out, out, x_mask, y_mask = g_model(pair_a, pair_b)

            batch_time.update(time.time() - end)
            end=time.time()

            progress.display(batch_index)


            if save_image:
                out = utils.transTensor2Numpy(out)
                x_out = utils.transTensor2Numpy(x_out)
                y_out = utils.transTensor2Numpy(y_out)
                x_mask = utils.transTensor2Numpy(x_mask)
                y_mask = utils.transTensor2Numpy(y_mask)
                cv2.imwrite(config["TEST"]["SAVE_IMAGE_DIR"]+f"x_out{batch_index}.png",x_out)
                cv2.imwrite(config["TEST"]["SAVE_IMAGE_DIR"]+f"y_out{batch_index}.png",y_out)
                cv2.imwrite(config["TEST"]["SAVE_IMAGE_DIR"]+f"out{batch_index}.png",out)
                cv2.imwrite(config["TEST"]["SAVE_IMAGE_DIR"]+f"x_mask{batch_index}.png",x_mask)
                cv2.imwrite(config["TEST"]["SAVE_IMAGE_DIR"]+f"y_mask{batch_index}.png",y_mask)
            batch_index += 1
            batch_data_a = input_img_a.next()
            batch_data_b = input_img_b.next()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path",
                        type=str,
                        default="./checkpoint_x2/FAP_SRE_Fuse_x2_Lytro.yaml",
                        required=True,
                        help="Path to test config file.")
    args = parser.parse_args()
    with open(args.config_path, "r") as f:
        cfg = yaml.full_load(f)

    device=torch.device("cuda",cfg["DEVICE_ID"])

    test_pair_a,test_pair_b = load_dataset(cfg, device)
    test_model=build_model(cfg,device)

    test(test_model,test_pair_a,test_pair_b,cfg)


