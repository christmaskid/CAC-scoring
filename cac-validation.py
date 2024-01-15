import os
import time
import torch
import numpy as np
from monai.networks.nets import SegResNetVAE, UNet
from monai.networks.layers import Norm
from monai.data import Dataset, DataLoader, decollate_batch
from monai.data.utils import pad_list_data_collate
from monai.transforms import (
	LoadImaged,
	EnsureChannelFirstd,
	Spacingd,
	ScaleIntensityRanged,
	Compose,
	AsDiscreted,
	Invertd,
	Activationsd,
	Invert,
	AsDiscrete,
)
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine
from calcium_scoring import get_all_calcium_scores, get_CVD_risk_category
import csv

device = torch.device("cuda:3")

configs = {'cac_labels': {1:"LM", 2:"LAD", 3:"LCx", 4:"RCA", 5:"other"}}

## MODEL: PASTE THE DESIRED ARCHITECTURE & PATH
model = UNet(
	spatial_dims=3,
	in_channels=1,
	out_channels=6,
	channels=(16, 32, 64, 128, 256),
	strides=(2, 2, 2, 2),
	num_res_units=2,
	norm=Norm.BATCH,
)
model.load_state_dict(torch.load("/home/b09401064/CAC/models/CAC_UNet_230820-002_best.pth")) # <-貼上要測試的model路徑
model.to(device)
model.eval()

## DATASET: PASTE THE DESIRED DIRECTORY
data_dir = "/data/b09401064/CAC_DATASET_ACL/" # <- 貼上要測試的檔案路徑；架構是資料夾底下有 images/ 和 labels/ ，可以調整。
data_list = []

for file in os.listdir(data_dir+"labels/"):
	if os.path.exists(data_dir+"images/"+file):
		data_list.append({
			"image-name": file,
			"image": os.path.join(data_dir, "images", file),
			"label": os.path.join(data_dir, "labels", file)
		})

transforms = Compose([
	LoadImaged(keys=("image", "label")),
	EnsureChannelFirstd(keys=("image", "label")),
	Spacingd(keys=("image"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
	ScaleIntensityRanged(keys=("image"), a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True)
])
post_transforms = Compose([
	Invertd(
		keys=("pred"),
		transform=transforms,
		orig_keys="image",
		meta_keys=("pred_meta_dict"),
		orig_meta_keys="image_meta_dict",
		meta_key_postfix="meta_dict",
		nearest_interp=True,
		to_tensor=True,
	),
	Activationsd(keys=("pred"), sigmoid=True), 
	AsDiscreted(keys="label", to_onehot=6), # PASTE THE DESIGNED NUMBER OF LABELS OF THE MODEL
])

dataset = Dataset(data=data_list, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=pad_list_data_collate)

## METRICS SETUP
dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

## TESTING LOOP
with torch.no_grad():
	for data in dataloader:
		data_name = data["image-name"][0].split(".")[0]
		print(data_name, flush=True)
		image, label = (data["image"].to(device), data["label"].to(device))

		roi_size = (96, 96, 96) # PASTE THE DESIGNED ROI SIZE OF THE MODEL
		sw_batch_size = 1
		data["pred"] = sliding_window_inference(image, roi_size, sw_batch_size, model)
		data = [post_transforms(i) for i in decollate_batch(data)]
		label, output = from_engine(["label", "pred"])(data)
		output[0] = output[0].cpu()

		# DEBUG
		# print(image.shape, label[0].shape, output[0].shape, image_ori.shape, flush=True)
		# torch.Size([1, 1, 200, 200, 127]) torch.Size([6, 512, 512, 43]) torch.Size([6, 512, 512, 43]) torch.Size([1, 512, 512, 43])

		dice_metric(y_pred=output, y=label) # include_background = False -> channel 0 (BG) is ignored

dice = dice_metric.aggregate()
print(f"Total: dice = {dice}", flush=True)


