# PU_Project_3

This is the PU Project of Maïc and Caius.


TODO:define the masses of every point 
TODO:somehow figure out the rotation around a central mass point
TODO: by generating 1000 random points in a circle two points can get very close to each other having an insane amount of energy
Idea: with every point generate a small dead zone or search a function that generates perfect circle and add some small randomness to get some energy in
TODO: if points are quick they can jump a collision because the collision happens during delta t
Idea: we use normal Euler(Krome) but instead of teleporting points we draw a line of p(i) to p(i+1) and if two lines get close we make delta t smaller for both points
Might be easier to just keep delta t small so that particles can't even get crazy accelerations (if decent start position there shouldn't be any problems) maybe also make pushingback_force a little weaker that they don't get crazy energys