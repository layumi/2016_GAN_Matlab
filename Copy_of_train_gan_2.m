function train_gan2(varargin)

% Load character dataset
imdb = load('./minist_data.mat') ;
imdb = imdb.imdb;
imdb.images.data = single(imdb.images.data)/255;%[0-1]
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

%netStruct = load('./data/pretrain_GAN/net-epoch-1.mat');
%net = dagnn.DagNN.loadobj(netStruct.net);
net = GDnet_2();
net.conserveMemory = false;
net.meta.averageImage = mean(imdb.images.data(:));

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = mean(imdb.images.data(:));
opts.train.batchSize = 128;
%opts.train.numSubBatches = 1 ;
opts.train.continue = false; 
opts.train.gpus = 4;
opts.train.prefetch = false ;
%opts.train.sync = false ;
%opts.train.errorFunction = 'multiclass' ;
opts.train.expDir = './data/GAN' ;
opts.train.learningRate = [0.01*ones(1,10)] ;
opts.train.derOutputs = {'Dobjective', 1,'Gobjective', 0} ;%%  this is defined in cnn_train_dag_gd2
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag_gd2(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
im = imdb.images.data(:,:,:,batch);% - opts.averageImage;
batchsize = numel(batch);
half = batchsize/2;
labels = [11*ones(1,half,'single'),imdb.images.label(batch(half+1:end))];  % 1 for data_rand     2 for data_gt
im_gt = im(:,:,:,half+1:end);
im_rand = rand(1,1,100,half,'single'); 
inputs = {'data_rand',gpuArray(im_rand),'data_gt',gpuArray(im_gt),'label',labels};