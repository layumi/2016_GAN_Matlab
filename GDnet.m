function net = GDnet()
net = dagnn.DagNN();
reluBlock = dagnn.ReLU('leak',0.1);
SigmoidBlock = dagnn.Sigmoid();
%G
conv1Block = dagnn.Conv('size',[3 3 1 64],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv1',conv1Block,{'data_rand'},{'G1'},{'G1f','G1b'});
net.addLayer('G_bn1',dagnn.BatchNorm(),{'G1'},{'G1bn'},{'Gbn1f','Gbn1b','Gbn1c'});
net.addLayer('Grelu1',reluBlock,{'G1bn'},{'G1x'});
conv2Block = dagnn.Conv('size',[1 1 64 32],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv2',conv2Block,{'G1x'},{'G2'},{'G2f','G2b'});
net.addLayer('G_bn2',dagnn.BatchNorm(),{'G2'},{'G2bn'},{'Gbn2f','Gbn2b','Gbn2c'});
net.addLayer('Grelu2',reluBlock,{'G2bn'},{'G2x'});
conv3Block = dagnn.Conv('size',[3 3 32 1],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv3',conv3Block,{'G2x'},{'G3'},{'G3f','G3b'});
net.addLayer('Grelu3',SigmoidBlock,{'G3'},{'G3x'});

%data
net.addLayer('concat',dagnn.Concat('dim',4),{'G3x','data_gt'},{'data'});

%D
conv4Block = dagnn.Conv('size',[5 5 1 20],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv1',conv4Block,{'data'},{'D1'},{'D1f','D1b'});
net.addLayer('D_bn1',dagnn.BatchNorm(),{'D1'},{'D1bn'},{'Dbn1f','Dbn1b','Dbn1c'});
net.addLayer('Drelu1',reluBlock,{'D1bn'},{'D1x'});
conv5Block = dagnn.Conv('size',[5 5 20 50],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv2',conv5Block,{'D1x'},{'D2'},{'D2f','D2b'});
net.addLayer('D_bn2',dagnn.BatchNorm(),{'D2'},{'D2bn'},{'Dbn2f','Dbn2b','Dbn2c'});
net.addLayer('Drelu2',reluBlock,{'D2bn'},{'D2x'});
conv6Block = dagnn.Conv('size',[4 4 50 100],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv3',conv6Block,{'D2x'},{'D3'},{'D3f','D3b'});
net.addLayer('D_bn3',dagnn.BatchNorm(),{'D3'},{'D3bn'},{'Dbn3f','Dbn3b','Dbn3c'});
net.addLayer('Drelu3',reluBlock,{'D3bn'},{'D3x'});
dropoutBlock = dagnn.DropOut('rate',0.5);
net.addLayer('dropout',dropoutBlock,{'D3x'},{'D3d'},{});
fcBlock = dagnn.Conv('size',[1 1 100 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('Dfc',fcBlock,{'D3d'},{'prediction'},{'Dfcf','Dfcb'});
lossBlock = dagnn.Loss('loss', 'softmaxlog');
net.addLayer('softmaxloss',lossBlock,{'prediction','label'},'objective');
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
net.addLayer('predict_label',dagnn.SoftMax(),'prediction','predict_label');
net.initParams();
end

