# Neuro Care
A Web App for Brain Tumor detection using MRI Scan
with user friendly interface:

### How to use:

- Click on **predict tumor** on right top corner. 

 ![1st](https://user-images.githubusercontent.com/75232316/141609886-b9f95387-9e6c-480c-bd5c-c7f78cd98e22.png)

- Upload the MRI scan in jpg format.

![5th](https://user-images.githubusercontent.com/75232316/141626668-17bf5774-f4ac-4074-a56b-0287c2fee542.png)

- Get the result within seconds with utmost accuracy.


![4th](https://user-images.githubusercontent.com/75232316/141612200-bd36f485-9864-44e5-ae7b-248719e253d6.png)


We also provide information regarding some famous Neurologists and an option to book an appointment respectively.

### How was it built:

We made a machine learning model which classifies the images using Convolutional Neural Network CNN.

By using a dataset from Kaggle, which comprises of both images with tumor and without tumor
then by splitting the dataset into training and testing sets with a ration of 80%:20%.

At the end, we used Sequential Model by kara to build our CNN model, and its respective layers to train the model
The Model gives the result with an accuracy level of 90%.

Have a look : https://paritosh-j.github.io/Neuro-Care/
