# Visual Summary Report

This report summarizes the key results from implementing ResNet-18 for image classification and a Transformer model for machine translation.

## Task 1: ResNet-18 on CIFAR-10

### Figure 1: Training and Validation Curves
![Loss and Accuracy Curves](../runs/cls/curves_cls.png)
*Caption: Training/validation loss decreases while accuracy increases, indicating successful learning. The model achieves >80% validation accuracy, meeting the acceptance criteria.*

### Figure 2: Confusion Matrix
![Confusion Matrix](../runs/cls/confusion_matrix.png)
*Caption: The normalized confusion matrix shows strong diagonal dominance, confirming high accuracy across most classes. Common confusion occurs between 'cat' and 'dog', and between 'truck' and 'car'.*

### Figure 3: Prediction Examples
![Correct Predictions](../runs/cls/preds_grid.png)
*Caption: A grid showing correctly classified images from the test set. The model confidently identifies objects across various classes.*

### Figure 4: Misclassification Examples
![Misclassified Predictions](../runs/cls/miscls_grid.png)
*Caption: A grid showing misclassified images. Errors often occur with ambiguous images or classes with high inter-class similarity (e.g., deer misclassified as bird in a non-canonical pose).*

### Figure 5: Grad-CAM Heatmaps
![Grad-CAM Heatmaps](../runs/cls/gradcam_results.png)
*Caption: Grad-CAM visualizations highlight the regions the model focuses on for prediction. The heatmaps align well with the objects of interest, confirming the model learned relevant features.*

---

## Task 2: Transformer for EN-DE Translation

### Figure 6: Training and Validation Loss Curves
![Loss Curves](../runs/mt/curves_mt.png)
*Caption: Both training and validation losses decrease steadily, indicating the Transformer model is effectively learning the translation task without significant overfitting.*

### Figure 7: Mask Visualization Demo
![Mask Visualization](../runs/mt/masks_demo.png)
*Caption: Visualization of a source padding mask (left, masking padding tokens at the end of the sequence) and a target causal mask (right, lower triangular matrix preventing attention to future tokens).*

### Figure 8: Attention Heatmap Example
![Attention Heatmap](../runs/mt/attention_layer1_head0.png)
*Caption: An attention heatmap from a decoder layer shows the alignment between source (German) and target (English) tokens. The diagonal alignment suggests successful word-to-word translation focus.*

### Figure 9: Translation Decode Examples and BLEU Score
| ![Decode Table](../runs/mt/decodes_table.png) | ![BLEU Score](../runs/mt/bleu_report.png) |
| :---: | :---: |
| *Caption (Left): Comparison of source sentences, ground truth translations, and model-generated predictions. The model produces coherent and often accurate translations.* | *Caption (Right): The corpus BLEU score exceeds the target threshold of 15, confirming the model's translation quality.* |
