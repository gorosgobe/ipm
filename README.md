# ipm
 Individual Project MEng Computing: "Imitation Learning with Visual Attention for Robot Control"
 
# Structure
**scripts/**: Scripts to run series of training runs automatically, with default parameters

**lib/**: Main libraries/code

**test_demonstrations/**: Test trajectories

**scenes/**: Scene data for simulator

**evaluation_results/**: Evaluation results for networks trained in Evaluation section of report

**tests/**: Some simple tests

Others: 

scripts to train every method we developed for this project (and more that didn't make it!) are available at the root directory

**models** and **test_results** keep some of the models and test results of those models we trained, mainly at the beginning of the project. The main results for every network we trained can be found at **evaluation_results**.

# Setup
To be able to generate demonstration datasets and test controllers, you need CoppeliaSim and PyRep installed. You may need to modify the PYTHONPATH environment variable to get the imports working.
 
# Code from other repositories
**maml/**: from tristandeleu/pytorch-maml

**senet_pytorch/**: from moskomule/senet.pytorch (used in early stages of project, now unused)

**lib/stn/linearized_multisampling_release**: from vcg-uvic/linearized_multisampling_release

Note: code in **lib/dsae** contains code from my own repository gorosgobe/dsae-torch
