function train_gan_info(varargin)

% Load character dataset
imdb = load('./minist_data.mat') ;
imdb = imdb.imdb;
imdb.images.data = single(imdb.images.data)/255;%[0-1]
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------

net = GDnet_info();
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
opts.train.expDir = './data/infoGAN' ;
opts.train.learningRate = [0.003*ones(1,20)] ;
opts.train.derOutputs = {'Dobjective', 0,'Gobjective',0} ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag_gd2_info(net, imdb, @getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb, batch,opts)
batchsize = numel(batch);
half = round(batchsize/2);
label = [ones(1,half,'single'),2*ones(1,batchsize-half,'single')];
c = randi(10,1,half);
code = zeros(10,half);
for i=1:half
   code(c(i),i)=1; 
end
im_rand = randn(64,half,'single'); 
im_rand = cat(1,im_rand,code);
im_rand = reshape(im_rand,1,1,[],half);
label2 = cat(2,c,c);
if(numel(label2)<batchsize)
   label2 = cat(2,label2,0); 
end
% select batch again
batch = [];
for i=1:batchsize - half 
    batch(i) = rand_same_class(imdb,label2(i+half));
end
im_gt = imdb.images.data(:,:,:,batch);
im_gt = reshape(im_gt,1,1,784,[]);
inputs = {'data_rand',gpuArray(im_rand),'data_gt',gpuArray(im_gt),'label',label,'label2',label2};