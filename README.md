## Skin Lesion Segmentation

This is a complete implementation of a U-Net model for image segmentation. It starts by loading the metadata and images for the dataset. Then, it splits the data into training and testing sets. Next, it defines the U-Net model architecture and 
compiles it with the Adam optimizer and binary crossentropy loss function.To improve the model's performance, the code uses data augmentation with aggressive techniques such as rotation, translation, shear, and zoom. It also uses early stopping
to prevent overfitting and a learning rate scheduler to gradually decrease the learning rate during training.

After training the model, the code evaluates it on the test set and prints the test loss and accuracy. It also calculates the confusion matrix, classification report, ROC curve, and AUC score to get a comprehensive understanding of the model's 
performance. Finally, the code chooses a random sample from the test set and displays the original image, ground truth mask, and predicted mask to visualize the model's output.

Overall, this code snippet provides a well-structured and comprehensive implementation of a U-Net model for image segmentation. It can be used as a starting point for developing own image segmentation models.


