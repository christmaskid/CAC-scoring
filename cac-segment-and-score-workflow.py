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
from calcium_scoring import get_all_calcium_scores, get_CVD_risk_category, get_all_calcium_scores_vote, get_all_calcium_scores_onehot
import csv

device = torch.device("cuda:7")

configs = {'cac_labels': {1:"LM", 2:"LAD", 3:"LCx", 4:"RCA", 5:"other"}}


## OUTPUT FILE
model_name = "CAC_UNet_231227-001_best.pth"
output_file = open("CAC_result_test.csv", "w")
writer = csv.writer(output_file)
writer.writerow(["Model tested: "+model_name])

writer.writerow(["Series Name", "No. of slice", "Slice thickness(mm)"] + [str(key)+" seg. dice" for key in ["LM", "LAD", "LCx", "RCA"]] 
	+ [str(key) + ' CAC score' for key in ['total', "LM", "LAD", "LCx", "RCA"]] + ['Risk category', 'Inference Time (s)'])

# PASTE THE DESIGNED ROI SIZE OF THE MODEL
model_roi_size = (96, 96, 96)

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
model.load_state_dict(torch.load("/home/b09401064/CAC/models/"+model_name))
model.to(device)
model.eval()

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

## DATASET: PASTE THE DESIRED DIRECTORY
data_dir = "/data/student/CAC模型驗證/segmentation/"
data_list = []

image_folder_name = "100筆影像" # "images"
label_folder_name = "100筆標註" # "labels"
for file in os.listdir(data_dir+label_folder_name+"/"):
	# if os.path.exists(data_dir+"100筆影像/"+file.split('.')[0]+'.nii.gz'):
	# print(file, file.split('.')[0]+'.nii.gz', flush=True)
	data_list.append({
		"image-name": file,
		"image": os.path.join(data_dir, image_folder_name, file.split('.')[0]+'.nii.gz'),
		"image-ori": os.path.join(data_dir, image_folder_name, file.split('.')[0]+'.nii.gz'),
		"label": os.path.join(data_dir, label_folder_name, file)
	})
print(len(data_list), flush=True)
print(data_list[0], flush=True)

transforms = Compose([
	LoadImaged(keys=("image", "label", "image-ori")),
	EnsureChannelFirstd(keys=("image", "label")),
	Spacingd(keys=("image"), pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
	ScaleIntensityRanged(keys=("image"), a_min=-300, a_max=300, b_min=0.0, b_max=1.0, clip=True)
])
post_transforms = Compose([
	Invertd(
		keys=("struct","pred"),
		transform=transforms,
		orig_keys="image",
		meta_keys=("struct_meta_dict", "pred_meta_dict"),
		orig_meta_keys="image_meta_dict",
		meta_key_postfix="meta_dict",
		nearest_interp=True,
		to_tensor=True,
	),
	Activationsd(keys=("struct", "pred"), sigmoid=True), 
	AsDiscreted(keys="label", to_onehot=6), # PASTE THE DESIGNED NUMBER OF LABELS OF THE MODEL
])

dataset = Dataset(data=data_list, transform=transforms)
dataloader = DataLoader(dataset, batch_size=1, num_workers=1, collate_fn=pad_list_data_collate)

## METRICS SETUP
dice_metric_case = DiceMetric(include_background=False, reduction="mean_batch")

all_tp = {key: 0 for key in configs['cac_labels'].values()}
all_tp_fn = {key: 0 for key in configs['cac_labels'].values()}
all_tp_fp = {key: 0 for key in configs['cac_labels'].values()}

## TESTING LOOP
with torch.no_grad():
	for data in dataloader:
		data_name = data["image-name"][0].split(".")[0]
		print(data_name, flush=True)
		image, label, image_ori = (data["image"].to(device), data["label"], data["image-ori"])
		image_shape = image_ori[0].shape

		start_time = time.time()

		roi_size_struct = (96, 96, 96)
		sw_batch_size_struct = 1
		data["struct"] = sliding_window_inference(image, roi_size_struct, sw_batch_size_struct, model_struct)

		roi_size = model_roi_size
		sw_batch_size = 1

		data["pred"] = sliding_window_inference(new_inputs, roi_size, sw_batch_size, model)

		## IF ADD HU>130 CHANNEL:
		# add_inputs = torch.tensor((image>130).astype(int)).to(device)
		# new_inputs = torch.cat((image, add_inputs), dim=1).to(device)
		# data["pred"] = sliding_window_inference(new_inputs, roi_size, sw_batch_size, model)

		data = [post_transforms(i) for i in decollate_batch(data)]
		label, struct, output = from_engine(["label", "struct", "pred"])(data)


		# DEBUG
		# print(image.shape, label[0].shape, output[0].shape, image_ori.shape, struct[0].shape, flush=True)
		# torch.Size([1, 1, 200, 200, 127]) torch.Size([6, 512, 512, 43]) torch.Size([6, 512, 512, 43]) torch.Size([1, 512, 512, 43]) torch.Size([6, 512, 512, 43])

		struct_result = np.argmax(struct[0], axis=0).astype(int)
		heart_mask = (struct_result==1).astype(int) * (image_ori[0] >= 130).astype(int) 
		# print(np.sum(heart_mask), flush=True)

		n_labels = 5
		cac_result = np.zeros((n_labels+1, image_shape[0], image_shape[1], image_shape[2]))
		pred_class = np.argmax(output[0][1:], axis=0) + 1
		for i in range(0, n_labels+1):
			cac_result[i] = (pred_class == i).astype(int) * heart_mask
		
		affine = image_ori[0].affine
		spacing = [abs(affine[i][i].item()) for i in range(3)]
		argmax_cac_result = np.argmax(cac_result[:-1], axis=0)
		argmax_cac_result *= (argmax_cac_result<5).astype(int) # remove other 
		# print(argmax_cac_result.shape, flush=True)


		mid_time = time.time()

		# result = get_all_calcium_scores_vote(image_ori[0], spacing, argmax_cac_result,
		# 	cac_label_names={1:"LM", 2:"LAD", 3:"LCx", 4:"RCA"}, min_vol=3)

		# result = get_all_calcium_scores(image_ori[0], spacing, argmax_cac_result,
		# 	cac_label_names={1:"LM", 2:"LAD", 3:"LCx", 4:"RCA"})#, min_vol=3)

		result = get_all_calcium_scores_onehot(image_ori[0], spacing, cac_result[:-1],
			cac_label_names={1:"LM", 2:"LAD", 3:"LCx", 4:"RCA"})#, min_vol=3)

		risk = get_CVD_risk_category(result["total"])

		end_time = time.time()

		print("Inference time:", mid_time - start_time, flush=True)
		print("CAC scoring time:", end_time - mid_time, flush=True)

		cac_result = torch.tensor(np.stack([cac_result]))
		dice_metric_case(y_pred=cac_result, y=label) # include_background = False -> channel 0 (BG) is ignored
		dice_list = [item.item() for item in dice_metric_case.aggregate()]
		print(dice_list, flush=True)
		dice_metric_case.reset()

		cac_result = cac_result.numpy()
		for i in range(1,n_labels+1):
			tp_fn = np.sum(label[0][i])
			tp_fp = np.sum(cac_result[0][i])
			tp = np.sum(label[0][i] * cac_result[0][i])

			print(f"{configs['cac_labels'][i]}: label = {tp_fn}, pred = {tp_fp}, intersect = {tp}; ", 
				f"precision = {tp/tp_fp}, recall = {tp/tp_fn}, dice(f1) = {2*tp/(tp_fp+tp_fn)}", flush=True)

			all_tp[configs['cac_labels'][i]] += tp
			all_tp_fn[configs['cac_labels'][i]] += tp_fn
			all_tp_fp[configs['cac_labels'][i]] += tp_fp

		print(argmax_cac_result.shape, flush=True)
		print(spacing, flush=True)
		print(result, flush=True)
		print(risk, flush=True)

		writer.writerow([data_name, image_shape[2], spacing[2]] + dice_list[:-1] + list(result.values()) + [risk, end_time-start_time])


print(f"Overall (Flattened):", flush=True)
writer.writerows([[],['Flattened'], ['Vessel', 'Precision', "Recall", "Dice (F1)"]])

total_tp, total_tp_fp, total_tp_fn = 0, 0, 0

for key in ["LM", "LAD", "LCx", "RCA"]:
	precision = all_tp[key]/all_tp_fp[key]
	recall = all_tp[key]/all_tp_fn[key]
	dice = all_tp[key]*2/(all_tp_fn[key]+all_tp_fp[key])

	print(f"{key}: precision = {precision}, recall = {recall}, dice = {dice}", flush=True)
	total_tp += all_tp[key]
	total_tp_fn += all_tp_fn[key]
	total_tp_fp += all_tp_fp[key]

	writer.writerow([key, precision, recall, dice])

precision = total_tp / total_tp_fp
recall = total_tp / total_tp_fn
dice = total_tp*2/(total_tp_fn+total_tp_fp)
print(f"Total: precision = {precision}, recall = {recall}, dice = {dice}", flush=True)

writer.writerow(["Total", precision, recall, dice])

output_file.close()



