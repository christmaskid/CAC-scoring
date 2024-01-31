import torch
import torch.nn as nn
import os
import time
import sys
import numpy as np
import nibabel as nib
import warnings
import glob
import gc
import nibabel as nib
from calcium_scoring import get_all_calcium_scores, get_CVD_risk_category
import csv
from monai.utils import set_determinism
from monai.transforms import (
	LoadImaged,
	EnsureChannelFirstd,
	ScaleIntensityRanged,
	Compose,
	AsDiscrete,
	CopyItemsd,
	ThresholdIntensityd,
	ScaleIntensityRanged,
	ConcatItemsd,
	ToTensor,
	Activations,
)
from monai.data import CacheDataset, Dataset, DataLoader, decollate_batch
from monai.data.utils import pad_list_data_collate
from monai.networks.nets import SegResNet, UNet, AttentionUnet, UNETR, SwinUNETR,SegResNetVAE
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference,SlidingWindowInferer
from monai.handlers.utils import from_engine
from monai.networks import one_hot
from my_utils import my_sliding_window_inference 
# turn off warnings
warnings.filterwarnings("ignore")

# =================================== #

model_name = "____.pth" # PASTE model name
output_file = open("____.csv", "w") # FILL IN output file name
task_name = '' # FILL IN task name (for copy file)
model_type = "" # PASTE model architecture type ("UNet", "UNETR", "SegResNetVAE")
input_mode = "" # PASTE model input type ("only" (image only), "HU130" (add additional >130 mask), "with-cor-seg" (has coronary a. segmentation))
os.environ["CUDA_VISIBLE_DEVICES"]="" # FILL IN gpu number(s)
batch_size = 1

# =================================== #

import shutil
if not os.path.exists("records"):
	os.mkdir("records")
record_file_name = __file__.split(".")[0]+'-'+task_name+'.py'
shutil.copyfile(__file__, os.path.join("records", record_file_name))

# =================================== #

writer = csv.writer(output_file)
writer.writerow(["Model tested: "+model_name])

writer.writerow(["Series Name", "No. of slice", "Slice thickness(mm)"] + [str(key)+" seg. dice" for key in ["LM", "LAD", "LCx", "RCA"]] 
	+ [str(key) + ' CAC score (pred.)' for key in ['total', "LM", "LAD", "LCx", "RCA"]] + ['Risk category (pred.)']
	+ ['Inference Time (s)', 'Model time', 'Scoring time'])


# =================================== #

# DEFINE CONFIGURATIONS
configs = {
	"roi_size": (256,256, 32),
	#"roi_size": (96, 96, 96)
	"cac_labels": {1: "LM", 2: "LAD", 3: "LCx", 4: "RCA", 5: "other"},
	"train_batch_size": batch_size,
	"model_type": model_type,
	"input_mode": input_mode
}
configs["num_labels"] = len(configs["cac_labels"])

if configs['input_mode'] == "with-cor-seg":
	configs["additional_input_labels"] = {1: "LM", 2: "LAD", 3: "LCx", 4: "RCA"}

print("Configurations:", configs, flush=True)

# =================================== #

# GET DATA PATH
data_dir = "/data/student/CAC模型驗證/segmentation/"
val_data_list = []

image_folder_name = "100筆影像" # "images"
label_folder_name = "100筆標註" # "labels"
for file in os.listdir(data_dir+label_folder_name+"/"):
	if "CVAI-0151-20161130-1mm" in file: continue # discarded file
	val_data_list.append({
		"image-name": file,
		"image": os.path.join(data_dir, image_folder_name, file.split('.')[0]+'.nii.gz'),
		"image-ori": os.path.join(data_dir, image_folder_name, file.split('.')[0]+'.nii.gz'),
		"cac_label": os.path.join(data_dir, label_folder_name, file)
	})
print(len(val_data_list), flush=True)
print(val_data_list[0], flush=True)


# =================================== #

# DEFINE TRANSFORMS

if configs['input_mode'] == "HU130":
	load_transforms = [
		LoadImaged(keys=["image", "cac_label","image-ori"]),
		EnsureChannelFirstd(keys=("image","cac_label")),
		CopyItemsd(keys=("image"),names=["image_thres"]),
		ScaleIntensityRanged(keys=["image"], a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True),
		ThresholdIntensityd(keys=("image_thres"),threshold=129),
		ScaleIntensityRanged(keys=["image_thres"], a_min=20, a_max=100, b_min=0.0, b_max=1.0, clip=True),
		ConcatItemsd(keys=("image","image_thres"),name="image"),
	]
else: 
	load_transforms = [
			LoadImaged(keys=data_keys),
			EnsureChannelFirstd(keys=["image", "cac_label"]),
			Spacingd(keys=data_keys, pixdim=(1.0, 1.0, 1.0), mode=spacing_modes),
			ScaleIntensityRanged(keys=["image"], a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True)
	]

val_transforms = Compose(load_transforms)

post_pred = Compose([
	ToTensor(),
	Activations(softmax=True),
	AsDiscrete(argmax=True, to_onehot=configs['num_labels']+1)])
	#saveimage
post_label = Compose([
	ToTensor(),
	AsDiscrete(to_onehot=configs['num_labels']+1)])


# =================================== #
# DEFINE DATASETS AND DATALOADERS

# val_dataset = CacheDataset(data=val_data_list, transform=val_transforms, cache_rate=1.0, num_workers=4)
val_dataset = Dataset(data=val_data_list, transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=1, num_workers=1, collate_fn=pad_list_data_collate)


# =================================== #
# DEFINE MODEL
def get_model(configs):
	in_channels = 1
	if 'additional_input_labels' in configs:
		in_channels += len(configs['additional_input_labels'])
	if configs['input_mode'] == "HU130":
		in_channels = 2
	out_channels = configs['num_labels']+1
	model_type = configs['model_type']
	if model_type == "UNet":
		return UNet(
			spatial_dims=3,
			in_channels=in_channels,
			out_channels=out_channels,
			channels=(16, 32, 64, 128, 256),
			strides=(2, 2, 2, 2),
			num_res_units=2,
			norm=Norm.BATCH,
		)
	if model_type == "UNETR":
		return UNETR(
			in_channels=in_channels,
			out_channels=out_channels,
			img_size=configs['roi_size'],
			spatial_dims=3,
			feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, 
			pos_embed='conv', norm_name='instance'
		 )
	if model_type == "SegResNetVAE":
		return SegResNetVAE(
			input_image_size=configs['roi_size'],
			in_channels=in_channels,
			out_channels=out_channels,
			spatial_dims=3,
			# vae_nz=512
		 )
	print("Model type "+model_type+" is not defined.", flush=True)
	return None

model = get_model(configs)
if model is None:
	exit()

# SET DEVICE AND SET MODEL TO BE PARALLEL

device = torch.device("cuda:0")

model = nn.DataParallel(model)
model.load_state_dict(torch.load(model_name))
model.to(device)
model.eval()

# =================================== #

# DEFINE METRICS, LOSS FUNCTION AND OPTIMIZER

dice_metric = DiceMetric(include_background=False, reduction="none")
dice_metric_mean = DiceMetric(include_background=False, reduction="mean_batch")

#added
all_tp = {key: 0 for key in configs['cac_labels'].values()}
all_tp_fn = {key: 0 for key in configs['cac_labels'].values()}
all_tp_fp = {key: 0 for key in configs['cac_labels'].values()}

all_left_tp, all_left_tp_fn, all_left_tp_fp = 0, 0, 0 # for LM+LAD+LCx

print("Metric: DiceMetric", flush=True)

print("===============================================", flush=True)

# =================================== #

with torch.no_grad():
	# get=0
	for val_data in val_loader:
		data_name = val_data["image-name"][0].split(".")[0][-22:]
		print(data_name, flush=True)

		start_time = time.time()

		if configs['input_mode'] == "with-cor-seg":
			val_inputs, val_labels, val_add_inputs = (
				val_data["image"].to(device),
				val_data["cac_label"].to(device),
				val_data["corseg_label"].to(device)
			)
			new_val_inputs = torch.cat((val_inputs, val_add_inputs[:, 1:, :, :, :]), dim=1)
		else:
			val_inputs, val_labels ,image_ori= (
				val_data["image"].to(device),
				val_data["cac_label"].to(device),
				val_data["image-ori"].to(device),
			)
			new_val_inputs = val_inputs

		image_shape = image_ori[0].shape
		roi_size = configs['roi_size']
		sw_batch_size = 4

		if configs['model_type'] == "SegResNetVAE":
			val_outputs = my_sliding_window_inference(new_val_inputs, roi_size, sw_batch_size, model)
		else:
			val_outputs = sliding_window_inference(new_val_inputs, roi_size, sw_batch_size, model)

		val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
		val_labels = [post_label(i) for i in decollate_batch(val_labels)]

		affine = image_ori[0].affine
		spacing = [abs(affine[i][i].item()) for i in range(3)]

		argmax_cac_result = np.argmax(val_outputs[0][1:5+1], axis=0)
		argmax_cac_result *= (argmax_cac_result<5).astype(int) # remove other 

		mid_time = time.time()

		result = get_all_calcium_scores(image_ori[0], spacing, argmax_cac_result,
			cac_label_names={1:"LM", 2:"LAD", 3:"LCx", 4:"RCA"}, min_vol=3)
		risk = get_CVD_risk_category(result["total"])

		end_time = time.time()

		dice_metric(y_pred=val_outputs, y=val_labels)
		dice_metric_mean(y_pred=val_outputs, y=val_labels)
		dice_list = [item.item() for item in dice_metric_mean.aggregate()]
		print(dice_list, flush=True)
		dice_metric_mean.reset()

		def get_metrics(label_name, label, pred):
			tp_fn = np.sum(label)
			tp_fp = np.sum(pred)
			tp = np.sum(label*pred)

			print(f"{label_name}: label = {tp_fn}, pred = {tp_fp}, intersect = {tp}; ", 
				f"precision = {tp/tp_fp}, recall = {tp/tp_fn}, dice(f1) = {2*tp/(tp_fp+tp_fn)}", flush=True)

			return tp, tp_fn, tp_fp

		for i in range(1,5+1):
			tp, tp_fn, tp_fp = get_metrics(configs['cac_labels'][i], val_labels[0][i], val_outputs[0][i])
			all_tp[configs['cac_labels'][i]] += tp
			all_tp_fn[configs['cac_labels'][i]] += tp_fn
			all_tp_fp[configs['cac_labels'][i]] += tp_fp

		left_pred = np.sum(val_outputs[0][1:3+1], axis=0)
		left_label = np.sum(val_labels[0][1:3+1], axis=0)
		print(left_pred.shape, left_label.shape, flush=True)
		left_tp, left_tp_fn, left_tp_fp = get_metrics("Union LM+LAD+LCx", left_label, left_pred)
		all_left_tp += left_tp
		all_left_tp_fn += left_tp_fn
		all_left_tp_fp += left_tp_fp

		print(argmax_cac_result.shape, flush=True)
		print(spacing, flush=True)
		print(result, flush=True)
		print(risk, flush=True)
		print("time:", end_time - start_time, f"({mid_time - start_time} + {end_time - mid_time}) (s)", flush=True)

		writer.writerow([data_name, image_shape[2], spacing[2]] + dice_list[:-1] + list(result.values()) 
			+ [risk, end_time-start_time, mid_time-start_time, end_time-mid_time])

		# get+=1
		# if get == 1:
			# nib.save(nib.Nifti1Image(val_inputs[0].cpu().numpy(),val_inputs[0].affine.cpu().numpy()),"output_img.nii.gz")
			# nib.save(nib.Nifti1Image(val_labels[0].cpu().numpy(),val_labels[0].affine.cpu().numpy()),"output_label.nii.gz")
			# nib.save(nib.Nifti1Image(val_outputs[0].cpu().numpy(),val_inputs[0].affine.cpu().numpy()),"output_pred.nii.gz")
			
print(f"Overall (Flattened):", flush=True)
writer.writerows([[],['Flattened'], ['Vessel', 'Precision', "Recall", "Dice (F1)"]])

total_tp, total_tp_fp, total_tp_fn = 0, 0, 0

def get_all_metrics(key, tp, tp_fn, tp_fp):
	precision = tp / tp_fp
	recall = tp / tp_fn
	dice = 2*tp / (tp_fn+tp_fp)
	print(f"{key}: precision = {precision}, recall = {recall}, dice = {dice}", flush=True)
	return precision, recall, dice

for key in ["LM", "LAD", "LCx", "RCA"]:
	precision, recall, dice = get_all_metrics(key, all_tp[key], all_tp_fn[key], all_tp_fp[key])

	total_tp += all_tp[key]
	total_tp_fn += all_tp_fn[key]
	total_tp_fp += all_tp_fp[key]

	writer.writerow([key, precision, recall, dice])

precision_l, recall_l, dice_l = get_all_metrics("Union LM+LAD+LCx", all_left_tp, all_left_tp_fn, all_left_tp_fp)
writer.writerow(["(Union LM+LAD+LCx)", precision_l, recall_l, dice_l])

precision_t, recall_t, dice_t = get_all_metrics("Total", total_tp, total_tp_fn, total_tp_fp)
writer.writerow(["Total", precision_t, recall_t, dice_t])

output_file.close()
