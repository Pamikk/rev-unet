# Deep Learning Template
This is a code template to train and test deep learning frameworks, I will orignize all of my dl related codes into this structure.
## Template Code Structure for Deep Learning
  + Data
    + dataset annotation
    + dataset pre-processing/analyze
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