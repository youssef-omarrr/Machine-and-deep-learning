
|Term|What it Does|Effect on Size|
|---|---|---|
|**Convolution**|Applies filters (kernels) to extract features|Usually **reduces** width/height depending on stride & padding|
|**Pooling**|Downsamples by summarizing nearby values (e.g. max/avg)|**Reduces** width/height, keeps channels same|
|**Kernel (Filter)**|Small matrix (e.g. 3×3) that slides over input to detect patterns|Determines **receptive field size**|
|**Stride**|How far the kernel moves each step|Larger stride → **smaller output**|
|**Padding**|Adds zeros around the image edges|Keeps **output same size** as input (if “same” padding)|