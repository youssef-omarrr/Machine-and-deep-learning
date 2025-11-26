## DDPM

The DDPM (Denoising Diffusion Probabilistic Model) sampling formula describes the **reverse diffusion process**, which iteratively transforms **pure random noise** into a **clear, high-quality image** by subtracting a small amount of predicted noise at each step.

It estimates the mean ($\mu_\theta$) and variance ($\Sigma_\theta$) of the reverse transition probability $p_\theta(\mathbf{x}_{t-1} | \mathbf{x}_t)$ to generate the less noisy image $\mathbf{x}_{t-1}$ from the noisy image $\mathbf{x}_t$.

The core of the formula is:
$$\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \mathbf{\epsilon}_\theta(\mathbf{x}_t, t) \right) + \sqrt{\tilde{\beta}_t} \mathbf{z}$$
where:
* $\mathbf{x}_{t-1}$ is the image at the next (less noisy) step.
* $\mathbf{x}_t$ is the current noisy image.
* $\mathbf{\epsilon}_\theta(\mathbf{x}_t, t)$ is the **predicted noise** (output of your Generator $G$).
* $\alpha_t$ and $\bar{\alpha}_t$ are constants from the fixed noise schedule.
* $\mathbf{z}$ is a small amount of random noise added for stochasticity (except for the very last step).