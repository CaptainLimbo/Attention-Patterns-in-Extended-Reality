# Machine Learning over Eye Tracking Data

This is the sub-repo for ML training over the collected gaze data, using the mvts-transformer model. This is forked from the [original repo](https://github.com/gzerveas/mvts_transformer), and we keep the original [README](../mvts_transformer/README_mvts.md) here as well. Please refer to the original README if you want to define your own datasets. 

## GazeBaseVR Data Pre-Processing
We employ a public eye-tracking data collected in VR to pretrain our model. 
[paper link](https://www.nature.com/articles/s41597-023-02075-5)    [dataset](https://figshare.com/articles/dataset/GazeBaseVR_Data_Repository/21308391)


GazeBaseVR was collected from a diverse population of 407 college-aged participants, recorded up to six times each over a 26-month period, each time performing a series of five different ET tasks: (1) a vergence task, (2) a horizontal smooth pursuit task, (3) a video-viewing task, (4) a self-paced reading task, and (5) a random oblique saccade task. The data was recorded at 250Hz, including gaze directions and eye center positions for both eyes in the Unity coordinate system. Our [pre-procossing pipeline](src/datasets/gazebasevr_preprocessing.py) is composed of the following steps:
1. downsampling the data to 1/4 
2. convert gaze direction vectors from spherical to cartesian
3. add columns to the data as those we have for the Sudoku study, using placeholders to fill in blanks for those features. These features will not be masked for the pretraining task and therefore will not affect pretraining.

Note that the pretraining data was also split to training & validation sets for performance monitoring. 


## Sudoku Data Pre-Processing
Please refer to [sudoku_preprocessing.py](src/datasets/gazebasevr_preprocessing.py) for details of our data pre-processing pipeline. As mentioned in the paper, for VR data we downsampled it by half first to match the 60Hz frequency in AR, and eyes open amount in AR data were first normalized based on the baseline (easy) trial before used for training/testing. For label extraction, we treat all timestamps that the mascot was marked running as the MR period. The maximum window length was set as 120 frames, while we discard any window with size < 60 to avoid too short data entries. 

For example, if you want to extract the dataset for ACS prediction on the AR data using gaze directions, eyes open amount and gaze targets as features, you can run:
```
convert_raw_data("datasets/AR_samples", "datasets/Sudoku_Split_Time/AR_direction_amount_target/ACS", use_2D_direction=True, label=list(map(grouping_ACS, AR_ACS)), use_center=False, use_amount=True, use_target=True)
```
If want to extract the dataset for MR prediction, you can run
```
convert_raw_data("datasets/AR_samples", "datasets/Sudoku_Split_Time/AR_direction_center_target/Mascot", use_2D_direction=True, label=None, use_center=True, use_amount=False, use_target=True)
```

Note that the Sudoku data was split 8:1:1 into train, val and test sets. The split was conducted on each user's data but was gathered together later on. 

## Training
### Pretraining
For pretraining, we used task of mask-recovery, and exclude the features that not exist in the pretraining dataset when masking. You can run
```
python src/main.py --output_dir experiments --comment "Pretraining on GazebaseVR" --name gazebasevr-pretrain --records_file Classification_records.xls --data_class sudoku --data_dir datasets/$path_to_your_dataset --pattern train --val_pattern val --batch_size 128 --epochs 5 --lr 0.001 --optimizer RAdam --d_model 64 --pos_encoding learnable --exclude_feats 0,1,2
```
If only the three "gaze targets" features are used when pretraining. If eye center positions, eyes open amount are used in addition, change `--exclude_features` accordingly.

Each pretrained model was trained with batch size 128, learning rate 1e-3 for 5 epochs. 


### Sudoku
Two options can be applied to the Sudoku Task of ACS / MR prediction. If training from scratch, run:
```
CUDA_VISIBLE_DEVICES=1 python src/main.py --output_dir experiments --comment "Sudoku from Scratch" --name AR_Mascot_$dir --records_file Classification_records.xls --data_dir datasets/Sudoku_Split_Time/AR_$dir/Mascot --data_class sudoku --pattern train --val_pattern val --test_pattern test --epochs 60 --lr 1e-3 --optimizer RAdam --batch_size 32 --pos_encoding learnable --d_model 64 --task classification --key_metric accuracy 
```

If fine-tuning from a pretrained model, run:
```
CUDA_VISIBLE_DEVICES=1 python src/main.py --output_dir experiments --comment "Sudoku from Scratch" --name AR_Mascot_$dir --records_file Classification_records.xls --data_dir datasets/Sudoku_Split_Time/AR_$dir/Mascot --data_class sudoku --pattern train --val_pattern val --test_pattern test --epochs 60 --lr 1e-3 --optimizer RAdam --batch_size 32 --pos_encoding fixed --d_model 64 --task classification --key_metric accuracy --change_output --load_model experiments/gazebasevr_pretrained_5cat_dim64_$dir/checkpoints/model_best.pth
```

Note that we train for 60 epochs with batch size 32 and learning rate 1e-3. For training from scratch we set the positional encoding to be learnable, while for pretrained models we made it fixed as it was proved to be better. 
