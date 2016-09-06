netStruct = load('./data/GAN/net-epoch-34.mat');
net = dagnn.DagNN.loadobj(netStruct.net);
net.mode = 'test';
net.move('gpu');
net.conserveMemory = false;
%im_mean = net.meta.normalization.averageImage;
im = rand(1,1,100,2,'single')-0.5;
net.eval({'data_rand',gpuArray(im)});
result = gather(net.vars(net.getVarIndex(('G3x'))).value);% + net.meta.averageImage;

figure(1);
hold on;
subplot(1,2,1);
imshow(imresize(result(:,:,:,1),1));
subplot(1,2,2);
imshow(imresize(result(:,:,:,2),1));