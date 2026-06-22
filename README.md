# NN_trial
A simple neural network with tanh activation functions, built up from scratch and compared with equivalent PyTorch code. Written for myself, for pedagogical purposed. Nothing new by itself.

The file NN_trial.py contains the definition of the NN_trial class. In the notebook NN_trial_nb.ipynb I use the class and compare it with a standard PyTorch routine. 
I test both approaches on some simple data, sampling sin(x) on the [0, pi] interval.
At this stage, one sees the much improved convergence of the Adam optimizer compared to the rudimentary gradient descent that I implemented (with some learning rate and some momentum decay rate).