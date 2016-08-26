function train_baseline(varargin)

% Load character dataset
imdb = load('./minist_data.mat') ;
imdb = imdb.imdb;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = GDnet();
net.conserveMemory = true;


% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = mean(imdb.images.data_mean(:));
opts.train.batchSize = 2;
%opts.train.numSubBatches = 1 ;
opts.train.continue = true; 
opts.train.gpus = 3;
opts.train.prefetch = false ;
%opts.train.sync = false ;
%opts.train.errorFunction = 'multiclass' ;
opts.train.expDir = '/data/GAN' ;
opts.train.learningRate = [0.1*ones(1,100),0.01*ones(1,20)] ;
opts.train.derOutputs = {'objective', 1} ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
im = imdb.images.data(:,:,:,batch);
batchsize = numel(batch);
half = round(batchsize/2);
labels = [ones(1,half,'single'),2*ones(1,batchsize-half,'single')];  % 1 for data_rand     2 for data_gt
im_gt = bxsfun(@minus,im(:,:,:,batchsize-half),opts.train.averageImage);
im_rand = rand(28,28,1,half);
inputs = {'data_rand',gpuArray(im_rand),'data_gt',gpuArray(im_gt),'label',labels};