# Experiment on AMSGrad -- pytorch version

## AMSGrad: a new exponetial moving average variant
Adam method does not always converge to the optimal solution [1]. AMSGrad is a new variant of Adam with guaranteed convergence while preserving the partical benefits of ADAM. 
AMSGrad uses the maximum of all v_t until the present time step and normalizes the running average of the gradient. By doing this, AMSGrad always has a non-increasing step size.

## The new version of Adam in Pytorch 
Apply AMSGrad in pytorch is quite easy, for example:
```
optimizer = torch.optim.Adam([x], lr=learning_rate, betas=(0.9, 0.99), eps=1e-8, amsgrad=True)
```

If we set amsgrad = False, then it's the origin version of Adam.

## Synthetic Experiments: Online Learning

I tested online learing verion of the syntheic experiments showing in the AMSGrad paper. The objective function is Ft(x) = 1010x if t mod 101 = 1, otherwise Ft(x) = -10x, and the constraint is x = [-1, 1].
The optimal solution is x = -1.

So far I have not seen anyone done such test in pytorch and compare with the original AMSGrad paper. There is a test using TensorFlow [2], however,
the test does not consider the constraint x=[-1, 1]. In fact, I found it's quite tricky to dealing with the constraint in TensorFlow, because 
it is not easy to handle tf.cond (if else in TensorFlow).
![Learning rate 0.1](/images/On01.png)
![Learning rate 0.01](/images/On001.png)
![Learning rate 0.001](/images/On0001.png)


### What did I find?
The orginal ICLR paper does not give the information about the learning rate to generate their plots in Figure 1. I am not quite sure
they used the same learning rate when they compare Adam with AMSGrad. In my experiments, I tested different learning rates and calculated the average regret Rt/t and 
the value of the iterate xt. Those are what I found:
- Adam always converge to the suboptimal solution +1, and AMSGrad guarante to reach the optimal solution -1.
- **The convergance of AMSGrad is very slow at smaller learning rate**. Maybe there is room to improve the performance of AMSGrad.


## Synthetic Experiments: Stochastic optimization
The stochastic optimization objective function is Ft(x) = 1010x with probability, otherwise Ft(x) = -10x. It's easy to define the objective
function by using a variable generating from Bernoulli distribution [3]. In pytorch, we can define the function:
```
from torch.distributions import Bernoulli
Bern_exp = Bernoulli(0.01)

def ft_sto(x):
    
    r = Bern_exp.sample() 
    loss = (r*1010.0 + (1 - r) * (-10.0))*x
    return loss, r
```

Except to calculate the value of xt during the iteration,
I also calculated the averge regret Rt/t. I found the value of Rt/t for both Adam and RMSGrad method decrease at the begining but increase after a while. If we stop too early, then we may trap in a local minima if we use RMSGrad. Again, Adam does not converge to the optimal solution.

![Learning rate 0.001](/images/sto.png)


## Reference
[1] https://openreview.net/pdf?id=ryQu7f-RZ

[2] https://github.com/junfengwen/AMSGrad/blob/master/toy.ipynb

[3] https://colab.research.google.com/notebook#fileId=1xXFAuHM2Ae-OmF5M8Cn9ypGCa_HHBgfG

