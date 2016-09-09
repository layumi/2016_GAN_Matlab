clear;
netStruct = load('./data/infoGAN/net-epoch-20.mat');  %after 20 epoch
k = 10 ;
net1 = dagnn.DagNN.loadobj(netStruct.net);
net1.mode = 'test';
net1.move('gpu');
net1.conserveMemory = false;
c = 1:10;
code = zeros(10,k);
for i=1:k
   code(c(i),i)=1; 
end
im_rand = rand(64,k,'single')-0.5; 
im_rand = cat(1,im_rand,code);
im_rand = reshape(im_rand,1,1,[],k);

net1.eval({'data_rand',gpuArray(im_rand)});
result1 = gather(net1.vars(net1.getVarIndex(('G3x'))).value);

for i=1:k
    hold on;
    subplot(1,k,i);
    imshow(reshape(result1(:,:,:,i),28,28,1));
end