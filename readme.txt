
Abhishek Meena
Department of Electrical Engineering
IIT KANPUR

>>Kinematic control of a redundant manipulator using inverse-forward adaptive scheme<<<<

Reference paper is attached with the documents


#####################
 Designed a Kohonen-Self Organizing Map based network that can model inverse kinematics, i.e given Cartesian
position, make prediction in joint space. 
--Took a 2-D lattice of size 10x10 . 
--Range of joint space is given as -pi<theta1<pi; -pi<theta2<pi . 
--Heuristically tune learning rate and variance for distance measure. 
You are free to increase the lattice size as well.
 After training is over, the network will predict joint angles for reaching a desired
Cartesian position. Given this predicted joint position, compute the actual Cartesian
position that the manipulator end-effector has reached using the forward kinematics.
$#@%%%%%%%%%%%%%%%%%

 Tracked a circle of 1.5 m radius around the manipulator base.
 Plotted the result while showing kinematic configuration.
