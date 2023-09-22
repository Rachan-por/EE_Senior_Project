# EE_senior_project:
**Image Super Resolution Improvement Using Image Enhancement Model**

## Data Prepare (prepare_data.m)
* prepare_data.m is a matlab file to resize image to low resolution.
* Set path_original in line 8 of this file to the path of your images
* Modify scale_all in line 13 of this file to the scale that you want

## Training
Command to train is: \
```python train.py --train_dir PATH_TRAIN —timesteps TIMESTEPS --loss LOSS_FUNCTION --num_epochs EPOCHS --batch_size BATCH --lr LEARNING_RATE ``` 
* Loss function has 2 options l1 and l2 loss. You can modify the loss in loss_function.py

## Track Experiment
* Track experiment using [Weights & Biases](https://wandb.ai/site)
* To track the training process, you have to change w&b to your account by changing ```wandb.login(key='YOURS')``` in line 63 of train.py
