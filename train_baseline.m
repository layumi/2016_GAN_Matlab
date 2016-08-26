function train_baseline(varargin)

% Load character dataset
imdb = load('./scale_minist.mat') ;
imdb = imdb.imdb;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = resnet52_bs();
net.conserveMemory = true;


% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = mean(imdb.images.data_mean(:));
opts.train.batchSize = 36;
%opts.train.numSubBatches = 1 ;
opts.train.continue = true; 
opts.train.gpus = 3;
opts.train.prefetch = false ;
%opts.train.sync = false ;
%opts.train.errorFunction = 'multiclass' ;
opts.train.expDir = '/data/resnet_baseline' ;
opts.train.learningRate = [0.1*ones(1,100),0.01*ones(1,20)] ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
im = imdb.images.data(:,:,:,batch);
labels = imdb.images.label(:,batch);
batchsize = size(im,4);
im = bsxfun(@minus,im,opts.averageImage);
oim = zeros(227,227,3,batchsize,'single');
for i =1:batchsize
    x = randi(29);
    y = randi(29);
    temp = im(x:x+226,y:y+226,:,i);
    r = rand>0.5;
    if r
        oim(:,:,:,i) = temp;
    else oim(:,:,:,i) = filplr(temp);
    end
end
inputs = {'x0',gpuArray(oim),'label',labels};