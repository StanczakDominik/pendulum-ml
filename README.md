# Double pendulum: a machine learning experiment

As seen on [Wikipedia](https://en.wikipedia.org/wiki/Double_pendulum#Chaotic_motion), 
the simple double pendulum can exhibit chaotic motion, conditional on the proper
initial conditions. There's a beautiful graph on Wikipedia that shows this phenomenon.

The question that comes to mind is:

Given our ability to easily simulate this kind of setup numerically, can we teach a machine learning system to predict
1. whether the pendulum will flip (classification!),
2. how long it will take a pendulum to flip (regression!)?

For now, running `pendulum.py` will generate a `hdf5` file with time series data.
