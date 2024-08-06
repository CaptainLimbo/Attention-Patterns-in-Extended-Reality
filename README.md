# Attention-Patterns-in-Extended-Reality
This is the official code repo for our ISMAR '24 paper, *"Looking" into Attention Patterns in Extended Reality: An Eye Tracking-Based Study* by Zhehan Qu, Ryleigh Byrne and [Maria Gorlatova](https://maria.gorlatova.com/current-research/). The Arxiv version of the paper can be found [here](temp). The overview of our study is shown in Figure 1.
![Figure 1](setup.png)

If you are interested, please also check out the video demos of our AR and VR apps (click on the images to go to YouTube).

### AR Demo Video
[![Watch the video](https://img.youtube.com/vi/k2SwEOOD1SQ/maxresdefault.jpg)](https://www.youtube.com/watch?v=WvbAxP_ZsGw)

### VR Demo Video
[![Watch the video](https://img.youtube.com/vi/WvbAxP_ZsGw/maxresdefault.jpg)](https://www.youtube.com/watch?v=WvbAxP_ZsGw)


Please refer to the individual READMEs at [first order analysis](gaze_data_analysis/README.md), [machine learning](mvts_transformer/README_Sudoku.md) and [sudoku_hint_generator](mvts_transformer/README_Sudoku.md) for code descriptions. Unfortunately, due to our IRB protocol we cannot release the eye gaze data we collected. Sample data are provided in [dataset](dataset) to show the format. 

If you find this code or any idea in the paper useful, please consider citing:
```
@INPROCEEDINGS{qu2024looking,
  author={Qu, Zhehan and Byrne, Ryleigh and Gorlatova, Maria},
  booktitle={Proceedings of IEEE ISMAR}, 
  title={``Looking'' into Attention Patterns in Extended Reality: An Eye Tracking-Based Study}, 
  year={2024},
}
```