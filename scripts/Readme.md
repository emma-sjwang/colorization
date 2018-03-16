# colorization 

1. run `python ./script/train_unet.py` will use '0,1,2,3' gpu to train the model, If you want to change it, change line 12 in train_unet.py file.

file tree
You need to store your initial data in ./data/testSetPlaces205\_resize/data/. Otherwith, you need to change line 20 in train_unet.py file.
And the filelist txt file should be stored in ./data/train\_data\_list.txt and ./data/test\_data\_list.txt

model is saved in ./model/
log which be check in tensorboard is saved in ./log_tf/baseline/date/

**logs_path** is defined on line 25 in unet.py
**Load_weights_from** is defined  on line 27 in unet.py

2. run `python ./script/test_unet.py` will use gpu 0 to test the model.
Before test, we need to transfer initial model weights into single model weights. We run it automatically inside test_unet.py file.

You can change filelist path, model\_name , data\_path in the end of test\_unet.py.
Test results weill be saved to save_path = "./data/results/baseline/". You can also change it in line 70 in test\_unet.py.
