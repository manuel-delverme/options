# options

Given an huge set of options I can dynamically (bayesian optimization style) find a good subset of options.

###### questions:
1. what's the metric of fitness for each option?
2. how do I compare two options trained for a different number of iterations? 
3. using Qlearning, how do i initialize the Q values for the new option?

##### answers:
###### fitness as f(iteration,option) is sum(Q[:,option]) at that time and 
1. store all the past observations and evaluate two options over the falue for the same iteration  
1. fit an exponentially decaying value and use 5tau (gaussian process)

problems:
other options Qs don't influence Qvalue for the considered option, e.g. some other very good option doesn't allow for precise estimation of the considered option Qvalue (Marlos uses options for exploration so this should not be as bad (?) )

###### fitness time to episode conclusion f(option) = num\_ep

problems:
highly non linear
unsatble?
