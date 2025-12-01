# Training Metrics Summary

## Diff_loss (Diffusion Loss):
Measures how well the model learns to remove noise and reconstruct clean images.
> Lower is better.

## gen_adv_loss (Generator Adversarial Loss):
Measures how well the generator fools the discriminator into thinking fake images are real.
> Lower is better.

## Disc_loss (Discriminator Loss):
Measures how well the discriminator separates real images from fake ones.
> - Medium values are healthy. 
> - Too low = discriminator too strong.
> - Too high = discriminator too weak.

---