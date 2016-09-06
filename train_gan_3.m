function train_gan3(varargin)

% Load character dataset
imdb = load('./minist_data_only2.mat') ;
imdb = imdb.imdb;
imdb.images.data = single(imdb.images.data)/255;%[0-1]
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = GDnet_3();
net.conserveMemory = false;
net.meta.averageImage = mean(imdb.images.data(:));

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = mean(imdb.images.data(:));
opts.train.batchSize = 128;
%opts.train.numSubBatches = 1 ;
opts.train.continue = true; 
opts.train.gpus = 4;
opts.train.prefetch = false ;
%opts.train.sync = false ;
%opts.train.errorFunction = 'multiclass' ;
opts.train.expDir = './data/GAN' ;
opts.train.learningRate = [0.003*ones(1,20)] ;
opts.train.derOutputs = {'Dobjective', 0,'Gobjective',0} ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag_gd2(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
im = imdb.images.data(:,:,:,batch);
label = imdb.images.label(batch);
batchsize = numel(batch);
half = round(batchsize/2);
labels = [ones(1,half,'single'),1+label(1:batchsize-half)];
%label(half+1:end)];  % 1 for data_rand     2 for data_gt
im_gt = im(:,:,:,1:batchsize-half);
im_gt = reshape(im_gt,1,1,784,[]);
im_rand = rand(1,1,100,half,'single')-0.5; 
inputs = {'data_rand',gpuArray(im_rand),'data_gt',gpuArray(im_gt),'label',labels};