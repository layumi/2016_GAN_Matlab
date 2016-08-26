function net = GDnet()
net = dagnn.DagNN();
reluBlock = dagnn.ReLU('leak',0);
%G
conv1Block = dagnn.Conv('size',[3 3 1 64],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv1',conv1Block,{'data_rand'},{'G1'},{'G1f','G1b'});
net.addLayer('Grelu1',reluBlock,{'G1'},{'G1x'});
conv2Block = dagnn.Conv('size',[1 1 64 32],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv2',conv2Block,{'G1x'},{'G2'},{'G2f','G2b'});
net.addLayer('Grelu2',reluBlock,{'G2'},{'G2x'});
conv3Block = dagnn.Conv('size',[3 3 32 1],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv3',conv3Block,{'G2x'},{'G3'},{'G3f','G3b'});
net.addLayer('Grelu3',reluBlock,{'G3'},{'G3x'});

%data
net.addLayer('concat',dagnn.Concat(),{'G3x','data_gt'},{'data'});

%D
conv3Block = dagnn.Conv('size',[5 5 1 20],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]); %12*12
net.addLayer('Dconv1',conv3Block,{'data'},{'D1'},{'D1f','D1b'});
net.addLayer('Drelu1',reluBlock,{'D1'},{'D1x'});
conv4Block = dagnn.Conv('size',[5 5 20 50],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv2',conv4Block,{'D1x'},{'D2'},{'D2f','D2b'});
net.addLayer('Drelu2',reluBlock,{'D2'},{'D2x'});
conv5Block = dagnn.Conv('size',[4 4 50 100],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv3',conv5Block,{'D2x'},{'D3'},{'D3f','D3b'});
net.addLayer('Drelu3',reluBlock,{'D3'},{'D3x'});
dropoutBlock = dagnn.DropOut('rate',0.5);
net.addLayer('dropout',dropoutBlock,{'D3x'},{'D3d'},{});
fcBlock = dagnn.Conv('size',[1 1 100 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc',fcBlock,{'D3d'},{'prediction'},{'fcf','fcb'});
lossBlock = dagnn.Loss('loss', 'softmaxlog');
net.addLayer('softmaxloss',lossBlock,{'prediction','label'},'objective');
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction','label'}, 'top5err') ;
net.initParams();
end

