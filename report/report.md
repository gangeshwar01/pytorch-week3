
# Comprehensive Report: ResNet-18 and Transformer Implementation

### **1. Sources Consulted**

This project required synthesizing information from foundational academic papers, official documentation, and practical implementation guides.

**Task 1: ResNet-18 Implementation**

* **Primary Paper:** He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
    * **Use:** Understood the core philosophy behind residual learning, the structure of the `BasicBlock` (two 3x3 convolutions), and the design principles for identity vs. projection shortcuts (Options A, B, and C).
* **PyTorch Documentation:** Official documentation for `torch.nn.Module`, `torch.nn.Conv2d`, `torch.nn.BatchNorm2d`, and `torch.nn.AdaptiveAvgPool2d`.
    * **Use:** Referenced for correct parameter initialization (e.g., `in_channels`, `out_channels`, `kernel_size`, `stride`, `padding`) and understanding the expected input/output dimensions of each layer.
* **CIFAR-10 Adaptation Guides:** Various online tutorials and public GitHub repositories implementing ResNet for CIFAR-10.
    * **Use:** The original ResNet architecture in the paper starts with a 7x7 convolution and stride 2, designed for 224x224 ImageNet images. These sources provided the standard modification for 32x32 images: replacing the initial layer with a smaller 3x3 convolution and stride 1, and removing the initial max pooling layer to preserve spatial resolution.
* **Grad-CAM Paper and Tutorials:** Selvaraju, R. R., et al. (2016). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391).
    * **Use:** Understood the algorithm for generating class activation heatmaps. Practical tutorials were necessary to correctly implement PyTorch hooks (`register_forward_hook` and `register_backward_hook`) to extract intermediate feature maps and gradients.

**Task 2: Transformer Implementation**

* **Primary Paper:** Vaswani, A., et al. (2017). *Attention Is All You Need*. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762).
    * **Use:** Provided the complete architectural blueprint, including multi-head attention, scaled dot-product attention formula, position-wise feed-forward networks, layer normalization placement, and the sinusoidal positional encoding equations.
* **Implementation Tutorial:** "The Annotated Transformer" by Harvard NLP (Sasha Rush).
    * **Use:** Served as an invaluable guide for structuring the code from scratch. It clarifies potential ambiguities in the paper by providing direct code equivalents for concepts like masking, cloning layers, and managing tensor shapes during attention calculation.
* **PyTorch Sequence-to-Sequence Tutorials:** Official PyTorch tutorials on TorchText and sequence-to-sequence modeling.
    * **Use:** Learned best practices for data preprocessing, including building vocabularies, tokenizing sentences (using libraries like SpaCy), handling special tokens (`<pad>`, `<bos>`, `<eos>`), and creating batches with padding using a custom `collate_fn`.

---

### **2. Key Learnings, Insights, and Conclusions**

#### **ResNet-18 Learnings**

* **Insight: The Power of the Identity Shortcut.** The core idea of ResNet is deceptively simple: `y = F(x) + x`. Implementing this revealed its profound effect on training stability. By allowing gradients to flow directly through the identity connection, ResNet avoids the vanishing gradient problem common in very deep "plain" networks. This allows for training significantly deeper models without degradation in performance.
* **Challenge: Dimension Matching for Projection Shortcuts.** The most critical implementation detail was handling the shortcut connection when downsampling occurs or channel dimensions change (e.g., moving from a block with 64 filters to one with 128 filters). The shortcut path `x` must be transformed to match the output shape of the convolutional path `F(x)`. This required implementing a *projection shortcut*: a 1x1 convolution with a stride of 2. Forgetting to match the stride in the shortcut path while downsampling in the main path results in a runtime tensor shape mismatch error.
* **Conclusion: Architecture Adaptation is Non-Trivial.** A model architecture is not one-size-fits-all. The initial layers of ResNet designed for ImageNet (224x224) are too aggressive for CIFAR-10 (32x32). The initial 7x7 convolution with stride 2 would reduce the feature map from 32x32 to 16x16, and the subsequent max pooling would reduce it further to 8x8 before the main residual blocks even begin. This excessive information loss would cripple performance. The adaptation (using a single 3x3 convolution with stride 1) was essential for achieving high accuracy on the smaller dataset.

#### **Transformer Learnings**

* **Challenge: The Primacy of Masking.** Implementing the Transformer from scratch highlighted that attention mechanisms are entirely dependent on correct masking. Two types of masks were critical and difficult to implement correctly:
    1.  **Padding Mask:** Prevents the model from paying attention to `<pad>` tokens in the input sequence. This is applied in both encoder self-attention and decoder cross-attention.
    2.  **Causal (Look-ahead) Mask:** Prevents positions in the decoder from attending to subsequent positions. This enforces autoregressive behavior, ensuring the prediction for token `i` only depends on tokens `0` to `i-1`.
    * **Insight:** The challenge lies in tensor broadcasting. A mask must often be expanded from `[batch_size, seq_len]` to `[batch_size, 1, 1, seq_len]` (for padding masks) or `[batch_size, 1, seq_len, seq_len]` (for causal masks) to interact correctly with the attention score matrix, which has shape `[batch_size, num_heads, seq_len, seq_len]`. Debugging these mask shapes was the most time-consuming part of the implementation.
* **Insight: The Role of Layer Normalization.** Unlike ResNet, which primarily uses Batch Normalization, the Transformer uses Layer Normalization. This normalizes features across the embedding dimension for each token independently, rather than across the batch. This design choice is critical for NLP tasks where sequence lengths vary, making batch statistics less reliable. The placement of LayerNorm *before* the sublayer (pre-norm), although differing from the original paper (post-norm), is a common technique that leads to more stable training.
* **Conclusion: Scaled Dot-Product Attention Mechanics.** The scaling factor $\frac{1}{\sqrt{d_k}}$ in the attention formula $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ is vital. Without it, for large values of $d_k$ (embedding dimension per head), the dot product values grow large in magnitude. This pushes the softmax function into regions where its gradient approaches zero, causing training to stall. This small detail is a perfect example of how subtle mathematical refinements enable deep learning architectures to train effectively.

---

### **3. Practice Attempts and Exercises**

To ensure correctness before integrating components into the final models, several isolated unit tests were performed.

* **ResNet Block Test Script:**
    * **Objective:** Verify that the `BasicBlock` module correctly handles both identity and projection shortcuts and produces outputs of the expected dimensions.
    * **Procedure:** A small script instantiated a single `BasicBlock`. Two test cases were run:
        1.  **Identity Case:** Input tensor of shape `[N, 64, 32, 32]` passed through a block with `stride=1` and `out_channels=64`. Verified that the output shape remained `[N, 64, 32, 32]`.
        2.  **Projection Case:** Input tensor of shape `[N, 64, 32, 32]` passed through a block with `stride=2` and `out_channels=128`. Verified that the output shape correctly downsampled to `[N, 128, 16, 16]`. This confirmed the logic for the 1x1 convolutional shortcut.

* **Transformer Mask Visualization:**
    * **Objective:** Confirm the logic for generating causal and padding masks before feeding them into the attention module.
    * **Procedure:** Created dummy input sequences containing padding indices. Wrote functions `create_padding_mask` and `create_causal_mask`. Printed the resulting binary tensors to visually inspect their structure. For the causal mask, confirmed it was a lower triangular matrix. For the padding mask, confirmed that `False` values corresponded precisely to the indices of `<pad>` tokens in the input. This visual debugging prevented complex shape-related errors during model training.

* **Multi-Head Attention Unit Test:**
    * **Objective:** Test the `MultiHeadAttention` module in isolation to ensure correct tensor manipulation for splitting heads and concatenating results.
    * **Procedure:** Created dummy `query`, `key`, and `value` tensors. Passed them through the module and checked two things:
        1.  **Output Shape:** Verified that the final output tensor shape was `[batch_size, seq_len, d_model]`, matching the input dimension.
        2.  **Masking Effect:** Applied a mask to specific positions. Extracted the attention weight matrix before the final multiplication with `V`. Verified that the softmax scores corresponding to masked positions were effectively zero, confirming the mask was applied correctly before the softmax operation.
