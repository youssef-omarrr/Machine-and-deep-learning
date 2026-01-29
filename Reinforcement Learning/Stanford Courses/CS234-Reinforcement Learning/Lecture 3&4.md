# Lecture 3
- Both vaalue and policy iterations are guarnteed to converge to optimal value function and a policy with an optimal value (assuming gamma < 1)
- Policy iteration max = A^s
- Value iteration max > A^s
- G_t : **return**, discounted sum of rewards from time step t to horizon 
- V(s)^pi : **state value function**, expected return starting in state s under policy pi
-  Q(s,a)^pi : **state-action value function**, expected return in state s, taking action a and following policy pi

- Using the esstimae of value function -> bootstraping
- Mote carlo policy evaluation 
	- Value function is just averaging returns
	- Doesnt have to be markov
	- 
- 