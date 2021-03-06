# Reversible U-net for image segmentation
+ This is a re-implementation of my course project for Medical Image Analysis based on paper:
+ Dataset: Hippocampus segmentation of Medical Segmentation Decathlon
  + I choose it because it is a 
## Code Structure for Deep Learning
  + Data
    + process_hdf5 save as hdf5
    + process_json(tbd) 
      + json output with images path and label path
  + Models
    + utils - necessary function
      + maybe move evaluation metric here?
    + network
    + loss
    + backbone network
    + network configuration
  + dataProcessing
    + dataloader for train and test
  + Utils
    + logger
    + evaluation/metric
    + non-maximum-supression
  + visualization - jupyter notebook
  + Trainer - APIs
    + save/load weights
    + lr scheduler
    + optimizer(sometimes need to pass in network for things like GAN)    
  + train
  + test
  + evaluation - evaluate predicted result
  + config - configurate parpameters