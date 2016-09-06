clear;
netStruct1 = load('./data/GAN/net-epoch-1.mat');  %after 1 epoch 
netStruct2 = load('./data/GAN/net-epoch-20.mat');  %after 20 epoch
k = 5 ;
net1 = dagnn.DagNN.loadobj(netStruct1.net);
net1.mode = 'test';
net1.move('gpu');
net1.conserveMemory = false;
im = rand(1,1,100,k,'single')-0.5;
net1.eval({'data_rand',gpuArray(im)});
result1 = gather(net1.vars(net1.getVarIndex(('G3x'))).value);

net2 = dagnn.DagNN.loadobj(netStruct2.net);
net2.mode = 'test';
net2.move('gpu');
net2.conserveMemory = false;
net2.eval({'data_rand',gpuArray(im)});
result2 = gather(net2.vars(net1.getVarIndex(('G3x'))).value);

for i=1:k
    hold on;
    subplot(2,k,i);
    imshow(reshape(result1(:,:,:,i),28,28,1));
end

for i=1:k
    hold on;
    subplot(2,k,k+i);
    imshow(reshape(result2(:,:,:,i),28,28,1));
end