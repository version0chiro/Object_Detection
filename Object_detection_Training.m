imageLabeler('./resize_image/')
%% load images dir
imageDir = fullfile(matlabroot,'resize_image');
%% load ground truth
data = load('labels.mat');
gTruth = data.gTruth;
%%

%% layer graph

inputImageSize = [300 300 3];
numClasses = 1;
network = 'resnet50';
featureLayer = 'activation_40_relu';
anchorBoxes = [64,64; 128,128; 192,192];
lgraph = fasterRCNNLayers(inputImageSize,numClasses,anchorBoxes, ...
                          network,featureLayer)

%%
options = trainingOptions('sgdm',...
          'InitialLearnRate',0.001,...
          'Verbose',true,...
          'MiniBatchSize',2,...
          'MaxEpochs',5,...
          'Shuffle','never',...
          'VerboseFrequency',1,...
          'CheckpointPath',tempdir,...
          'ExecutionEnvironment','gpu');
%%
[imds,bxds] = objectDetectorTrainingData(gTruth);
%%
cds = combine(imds,bxds);
%%
[detector,info] = trainFasterRCNNObjectDetector(cds,lgraph,options);
%%
S = dir('resize_image'); % all of the names are in the structure S anyway.
N = {S.name};
%% saving the variable
filename = 'savedVariable.mat';
save(filename)

%% load the trained varible with this
load('savedVarible.mat');

%%
for i=3:168
    I = imread('resize_image/'+string(N(i)));
    [bboxes,scores] = detect(detector,I);
    if(~isempty(bboxes))
    I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    figure
    imshow(I)
    end
end
%%

I = imread('resize_image/IMG_20210201_150524.jpg');
%%
gpuDevice 
%%
[bboxes,scores] = detect(detector,I);
%%
if(~isempty(bboxes))
  I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
end
figure
imshow(I)