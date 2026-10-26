PLAN 

- spatial math
- create the simulation architecture with easy to use api flexible for single arm and dual arm
- set up the libraries for forward, inverse kinematics pinnochio, pink
  
STUDY how sim_cyro is setup and try to replicate it, you have to know what is happening inside

- once these things are stable proceed to do dls, qp ik solvers, interesting jacobian experiments, motion profiles etc
  
this repository should let you test, experiment and analyse everything quickly
if you learn a concept i must be easy to write and test it


What is the plan for this week?
- spatial math finish
- study sim cyro, everyday an hr or two must be spent in understanding the structure
  you have to learn to write good code which can only come from the pros
- a flexible pipeline or atleast spawn the arm 
- study multi limb ik whats happening inside 


What is the plan for tomorrow?
- close the loop : add another term to the feedforward loop which is to correct any error in the static transform
- add bounds and limits so that it does not drift to limit/singularity