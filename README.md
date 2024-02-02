## testing
* ```testing_other_metrics.py```: testing_.py 的更完整版，包含 voxel-wise 的 confusion matrix、計算一些 class 的 union 後的表現。
  * command line arguments:
    * task_name: ```str
    * model_type: ```str | ["UNet", "UNETR", "SegResNetVAE"]```; architecture
    * input_mode: ```str | ["only", "HU130", "with-cor-seg"]```; image only / with HU>130 mask / with coronary artery segmentation
    * model_name: ```str```; path for testing model
    * csv_file_name: ```str```; file for output ("xxx.csv")
    * cuda visible devices: ```str```; ```os.environ["CUDA_VISIBLE_DEVICES"]```
    * use_struct_model: ```bool | [True, False]```; whether to use structural mask
    * added_keys: ```list of tuples```; evaluating united classes performance
    * min_vol: ```"None" / int```; min. volume of calcium lesion
    * spacing_and_invert: ```bool | [True, False]```; whether to resample in pre-transform & inverted in post-transform
    * inf_argmax_start: ```int (0, 1)```; 0 - include background / 1 - do not include background, in inference (usually 0, 1 only in using structural mask)
    * roi_size: ```tuple, dim = 3```
* Example usage:
  
  ```python testing_save.py 231228-001 SegResNetVAE HU130 /home/student/exercise/CAC_SegResNetVAE_SegResNetVAE-231228-001_chosen_dict.pth CAC_result_test_231228-001_testing_min0_save.csv 6 False "[(1,2,3),(1,2,3,4)]" None False 0 "(256,256,32)```
  
* ```testing_.py```: 完整testing，可設定不同transform、包含或不包含structural mask。
* ```cac-segment-and-scoring-workflow.py```: 舊版完整testing，包含structural mask。

## tools
* ```calcium_scoring.py```: 主要使用 ```get_all_calcium_scores()``` 。
* ```my_utils.py```
  * ```my_sliding_window_inference```: for MONAI SegResNetVAE to use
