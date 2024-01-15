from scipy import ndimage
import numpy as np

# ref: https://github.com/qurAI-amsterdam/calcium-scoring/blob/main/src/calciumscoring/scores.py

def get_all_calcium_scores(
		image, spacing, label,
		cac_label_names=None, 
		min_vol=None, max_vol=None,
		verbose=False
	):


	result = dict()

	# total score
	total_cac_label = (label>0).astype(int)
	total_score = get_agatston_score(image, spacing, total_cac_label, min_vol, max_vol)
	result["total"] = total_score

	if verbose:
		# print(np.sum(total_cac_label), flush=True)
		print("Total:", total_score, flush=True)

	# vessel-specific scores
	score_sum = 0.0 # for debug

	if cac_label_names is not None:
		for key in cac_label_names.keys():
			ves_spe_label = (label==key).astype(int)
			ves_spe_score = get_agatston_score(image, spacing, ves_spe_label, min_vol, max_vol)
			result[cac_label_names[key]] = ves_spe_score

			if verbose: 
				# print(np.sum(ves_spe_label), flush=True)
				print(f"{cac_label_names[key]}: {ves_spe_score}", flush=True)

			score_sum += ves_spe_score # for debug

		#for debug
		if abs(score_sum - total_score) > 1:
			print(f"Score wrong: sum = {score_sum}, != total score = {total_score}", flush=True)

	return result

def get_CVD_risk_category(agatston_score):
	if agatston_score == 0:
		return "very low"
	elif agatston_score <= 10:
		return "low"
	elif agatston_score <= 100:
		return "intermediate"
	elif agatston_score <= 400:
		return "high"
	else:
		return "very high"

def density_factor(maxHU):
	if maxHU < 130:
		return 0
	if maxHU < 200:
		return 1
	if maxHU < 300:
		return 2
	if maxHU < 400:
		return 3
	return 4

def get_agatston_score(image, spacing, labels, min_vol = None, max_vol = None):
	"""
	Eliminate small lesions by lesion volume.
	"""

	# Assume labels are binary masks in [x, y, z]
	lesion_map, n_lesion = ndimage.label(labels, ndimage.generate_binary_structure(3,3))
	# print(n_lesion, "lesion(s)", flush=True)
	agatston_score = 0.0
	# volume_score = 0.0

	for lesion_num in range(1, n_lesion+1):
		lesion_mask = (lesion_map == lesion_num).astype(int)
		lesion_volume = np.sum(lesion_mask)

		if min_vol is not None and lesion_volume < min_vol:
			continue
		if max_vol is not None and lesion_volume > max_vol:
			continue

		# volume_score += lesion_volume

		lesion_score = 0.0
		slices_num = sorted(np.unique(np.where(lesion_mask==1)[2]))
		for z in slices_num:
			area = np.sum(lesion_mask[:, :, z])
			maxHU = np.max(image[:, :, z] * lesion_mask[:, :, z])
			lesion_score += area * density_factor(maxHU)
		# print(lesion_score, flush=True)
			
		agatston_score += lesion_score

	agatston_score *= spacing[0] * spacing[1] * spacing[2]/3.0
	# volume_score *= spacing[0] * spacing[1] * spacing[2]
	# density_score = agatston_score / volume_score * spacing[2]

	# print(agatston_score, volume_score, density_score, flush=True)
	return agatston_score#, volume_score, density_score

