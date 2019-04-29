%Ryan O'Shea
%SVM image classifier for identifying 16QAM, 32QAM, and 64QAM
%Feature extraction and model training take a pretty long time so my Bag of
%features and classifier are saved as featureBag.mat and SVM.mat
%respectively

%Define directories where training data is stored
DataDir = fullfile('Data')
TestDir = fullfile('Testing')

% |imageDatastore| recursively scans the directory tree containing the images. Folder names are automatically used as labels for each image.
trainingSet = imageDatastore(DataDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%Create a figure to show examples of the 3 different classes
figure;

subplot(1,3,1);
imshow(trainingSet.Files{133}); %Files 1-275 are 16QAM
title('16QAM Example')

subplot(1,3,2);
imshow(trainingSet.Files{418}); %Files 276-550 are 32QAM
title('32QAM Example')

subplot(1,3,3);
imshow(trainingSet.Files{654}); %Files 551-825 are 64QAM
title('64QAM Example')


%Detect SURF features on one sample from each of the three different
%classes. Bag of features uses SURF feature detection to extract features
image16 = rgb2gray(readimage(trainingSet, 100));
points16 = detectSURFFeatures(image16);

image32 = rgb2gray(readimage(trainingSet, 400));
points32 = detectSURFFeatures(image32);

image64 = rgb2gray(readimage(trainingSet, 700));
points64 = detectSURFFeatures(image64);


%Create a figure to show examples of the 3 different classes with their
%features highlighted
figure;

subplot(1,3,1);
imshow(image16); hold on;
plot(points16.selectStrongest(10));
title('16QAM Features Example')

subplot(1,3,2);
imshow(image32); hold on;
plot(points32.selectStrongest(10));
title('32QAM Features Example')

subplot(1,3,3);
imshow(image64); hold on;
plot(points64.selectStrongest(10));
title('64QAM Features Example')

%Create a bag of features to be used by the image classifier. Uses SURF
%feature extraction
%featureBag = bagOfFeatures(trainingSet); %Uncomment this if the features
%need to be extraced again


%Create the model with the training data and features.
%trainImageCategoryClassifier returns a linear SVM to classify an
%image with the category that it most closely matches
%Category 1 = 16QAM
%Category 2 = 32QAM
%Category 3 = 64QAM
disp("Now training model....");
%SVM = trainImageCategoryClassifier(trainingSet, featureBag); %Uncomment
%this if the model needs to be trained again
disp("Model training finished!");

%Testing data to verify the model after training. Took 25 samples from each
%class and placed them in a new testing dataset
testingSet = imageDatastore(TestDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%Use a confusion matrix to evaluate the classifier
confusionMatrix = evaluate(SVM,testingSet);

%Returns a 100% accuracy across the board

%To use the trained network to classify an image call predict(SVM,
%readimage("YourDataStore, index)). Use imread() if using a local file


