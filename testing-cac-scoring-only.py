import torch
import torch.nn as nn
import os
import time
import sys
import numpy as np
import pandas as pd
import sys

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
	ToTensord,
	Activations,
	Invertd,
	Activationsd,
	Spacingd,
	AsDiscreted
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

print(sys.argv, flush=True)

task_name = '231208-new2_no-rot'
model_type = "UNet"
input_mode = "only"
model_name = "/home/cvai_607/radiology_CAC/model/segmentation_new2_cac_no_rotation.pt"
output_file = open("CAC_result_test_231208-new2_no-rot_testing_dicom.csv", "w")
os.environ["CUDA_VISIBLE_DEVICES"]="4"
batch_size = 1
use_struct_model = True
min_vol = None
spacing_and_invert = True
inf_argmax_start = 1
roi_size = (96,96,96)

print(task_name, model_type, input_mode, model_name, output_file, batch_size, use_struct_model, flush=True)

# =================================== #

import shutil
if not os.path.exists("records"):
	os.mkdir("records")
record_file_name = __file__.split(".")[0]+'-'+task_name+'.py'
shutil.copyfile(__file__, os.path.join("records", record_file_name))

# =================================== #

writer = csv.writer(output_file)
writer.writerow(["Model tested: "+model_name])

writer.writerow(["Series Name", "No. of slice", "Slice thickness(mm)"] 
	+ [str(key) + ' CAC score (pred.)' for key in ['total', "LM", "LAD", "LCx", "RCA"]] + ['Risk category (pred.)']
	+ ['Inference Time (s)', 'Model time', 'Scoring time'])


# =================================== #

# DEFINE CONFIGURATIONS
configs = {
	"roi_size": roi_size,
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
data_dir = "/data/student/CAC模型驗證/dicom_test/"
val_data_list = []

for subdir in os.listdir(data_dir):
	print(subdir, flush=True)
	se_dir = os.listdir(os.path.join(data_dir, subdir))[0]
	print(se_dir, flush=True)
	image_path = os.path.join(data_dir, subdir, se_dir)
	print(image_path, flush=True)
	item = {
		"image-name": subdir, 
		"image": image_path,
		"image-ori": image_path,
	}
	if use_struct_model:
		item["image2"] = image_path,
	val_data_list.append(item)

val_data_list = sorted(val_data_list, key=lambda x: x["image-name"])

# val_data_list = val_data_list[94:] #debug

print(len(val_data_list), flush=True)
print(val_data_list[0], flush=True)


# =================================== #

# DEFINE TRANSFORMS

if spacing_and_invert == True: # Yu-Tong's model

	val_transforms = Compose([
		LoadImaged(keys=("image", "image-ori")),
		EnsureChannelFirstd(keys=("image")),
		Spacingd(keys=("image"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
		ScaleIntensityRanged(keys=("image"), a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True)
	])

else: # Li-Chueh's model
	image_keys = ["image"]
	if use_struct_model:
		image_keys += ["image2"]

	load_transforms = [
			LoadImaged(keys=image_keys+["image-ori"]),
			EnsureChannelFirstd(keys=image_keys)
	]

	if configs['input_mode'] == "HU130":
		load_transforms += [
			CopyItemsd(keys=("image"),names=["image_thres"]),
			ScaleIntensityRanged(keys=image_keys, a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True),
			ThresholdIntensityd(keys=("image_thres"),threshold=129),
			ScaleIntensityRanged(keys=["image_thres"], a_min=20, a_max=100, b_min=0.0, b_max=1.0, clip=True),
			ConcatItemsd(keys=("image","image_thres"),name="image"),
		]
	else: 
		load_transforms += [
			ScaleIntensityRanged(keys=image_keys, a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True)
		]

	val_transforms = Compose(load_transforms)


output_keys = ("struct", "pred") if use_struct_model else ("pred")
post_transforms = [
	Activationsd(keys=output_keys, sigmoid=True),
	AsDiscreted(keys="pred", argmax=True, to_onehot=configs['num_labels']+1),
	# ToTensord(keys=output_keys),
]
if spacing_and_invert:
	post_transforms = [
		Invertd(
			keys=output_keys,
			transform=val_transforms,
			orig_keys="image",
			meta_keys=(key+"_meta_dict" for key in output_keys),
			orig_meta_keys="image_meta_dict",
			meta_key_postfix="meta_dict",
			nearest_interp=True,
			to_tensor=True,
		),
	] + post_transforms
post_transforms = Compose(post_transforms)

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

if "dict" in model_name:
	model = nn.DataParallel(model)
model.load_state_dict(torch.load(model_name))
model.to(device)
model.eval()

if use_struct_model:
	model_struct = UNet(
		spatial_dims=3,
		in_channels=1,
		out_channels=6,
		channels=(16, 32, 64, 128, 256),
		strides=(2, 2, 2, 2),
		num_res_units=2,
		norm=Norm.BATCH,
	)
	model_struct_name = "models/CAC_unet_struct_160_20221224.pth"
	print("struct model: "+model_struct_name, flush=True)
	model_struct.load_state_dict(torch.load(model_struct_name))
	model_struct.to(device)
	model_struct.eval()

# =================================== #

print("===============================================", flush=True)

# =================================== #

with torch.no_grad():
	# get=0
	for val_data in val_loader:
		data_name = val_data["image-name"][0].split(".")[0][-22:]
		print(data_name, flush=True)

		start_time = time.time()

		# ===================# 
		
		val_inputs, image_ori = (
			val_data["image"].to(device),
			val_data["image-ori"].to(device),
		)
		if configs['input_mode'] == "with-cor-seg":
			new_val_inputs = torch.cat((val_inputs, val_add_inputs[:, 1:, :, :, :]), dim=1)
		else:
			new_val_inputs = val_inputs
		if use_struct_model:
			if spacing_and_invert:
				val_inputs2 = val_inputs
			else:
				val_inputs2 = val_data["image2"].to(device)

		# ===================# 
		
		image_shape = image_ori[0].shape
		roi_size = configs['roi_size']
		sw_batch_size = 4


		if configs['model_type'] == "SegResNetVAE":
			val_data["pred"] = my_sliding_window_inference(new_val_inputs, roi_size, sw_batch_size, model)
		else:
			val_data["pred"] = sliding_window_inference(new_val_inputs, roi_size, sw_batch_size, model)

		if use_struct_model:
			val_data["struct"] = sliding_window_inference(val_inputs2, (96, 96, 96), 4, model_struct)

		val_data = [post_transforms(i) for i in decollate_batch(val_data)]
		val_outputs =  from_engine(("pred"))(val_data)
		if use_struct_model:
			val_structs = from_engine(("struct"))(val_data)

		print(val_inputs.shape, val_outputs[0].shape, end=' ', flush=True)
		if use_struct_model: print(val_structs[0].shape, end='', flush=True)
		print()

		# ===================# 
		
		if use_struct_model:
			struct_result = np.argmax(val_structs[0], axis=0).astype(int)
			struct_mask = (struct_result==1).astype(int) #+ (struct_result==2).astype(int)
			HU_mask = (image_ori[0] >= 130).astype(int) 
			heart_mask = (struct_mask>0).astype(int) * HU_mask
			heart_mask = torch.Tensor(heart_mask).to(device)

			# for i in range(1, 5+1):
			# 	print(i, ", ", np.sum(val_outputs[0][i]==1), end='; ', flush=True)
			# print()
			# val_outputs[0] = torch.Tensor(val_outputs[0] * heart_mask)
			# print(val_outputs[0].shape, flush=True)
			for i in range(1, 5+1):
				val_outputs[0][i] *= heart_mask
				# print(i, ", ", np.sum(val_outputs[0][i]==1), end='; ', flush=True)
			# print()

		# ===================# 

		argmax_cac_result = np.argmax(val_outputs[0][inf_argmax_start:], axis=0)
		argmax_cac_result *= (argmax_cac_result<5).astype(int) # remove other 
		# for i in range(1, 5+1):
		# 	print(i, ", ", np.sum(argmax_cac_result==i), end='; ', flush=True)
		# print()
		
		mid_time = time.time()

		affine = image_ori[0].affine
		spacing = [abs(affine[i][i].item()) for i in range(3)]
		result = get_all_calcium_scores(image_ori[0], spacing, argmax_cac_result,
			cac_label_names={1:"LM", 2:"LAD", 3:"LCx", 4:"RCA"}, min_vol=min_vol)
		risk = get_CVD_risk_category(result["total"])

		# ===================# 
		
		end_time = time.time()

		# ===================# 
		
		print(argmax_cac_result.shape, flush=True)
		print(spacing, flush=True)
		print(result, flush=True)
		print(risk, flush=True)
		print("time:", end_time - start_time, f"({mid_time - start_time} + {end_time - mid_time}) (s)", flush=True)

		# ===================# 
		
		writer.writerow([data_name, image_shape[2], spacing[2]] # + dice_list 
			+ list(result.values()) + [risk, end_time-start_time, mid_time-start_time, end_time-mid_time])

		# ===================# 
		
		# get+=1
		# if get == 1:
		# nib.save(nib.Nifti1Image(val_inputs[0].cpu().numpy(),val_inputs[0].affine.cpu().numpy()),"output_img.nii.gz")
		# nib.save(nib.Nifti1Image(heart_mask.cpu().numpy(),val_inputs[0].affine.cpu().numpy()),"output_mask.nii.gz")
		# nib.save(nib.Nifti1Image(val_labels[0].cpu().numpy(),val_labels[0].affine.cpu().numpy()),"output_label.nii.gz")
		# nib.save(nib.Nifti1Image(val_outputs[0].cpu().numpy(),val_inputs[0].affine.cpu().numpy()),"output_pred.nii.gz")

		del val_data, argmax_cac_result, val_outputs, val_inputs
		if use_struct_model: del val_structs, struct_result, heart_mask, struct_mask
		gc.collect()
			
output_file.close()


