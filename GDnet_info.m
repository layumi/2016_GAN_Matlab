function net = GDnet_info()
net = dagnn.DagNN();
GreluBlock = dagnn.ReLU('leak',0);
reluBlock = dagnn.ReLU('leak',0.2);
SigmoidBlock = dagnn.Sigmoid();
%G
conv1Block = dagnn.Conv('size',[1 1 74 500],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv1',conv1Block,{'data_rand'},{'G1'},{'G1f','G1b'});
net.addLayer('G_bn1',dagnn.BatchNorm(),{'G1'},{'G1bn'},{'Gbn1f','Gbn1b','Gbn1c'});
net.addLayer('Grelu1',GreluBlock,{'G1bn'},{'G1x'});
conv2Block = dagnn.Conv('size',[1 1 500 500],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv2',conv2Block,{'G1x'},{'G2'},{'G2f','G2b'});
net.addLayer('G_bn2',dagnn.BatchNorm(),{'G2'},{'G2bn'},{'Gbn2f','Gbn2b','Gbn2c'});
net.addLayer('Grelu2',GreluBlock,{'G2bn'},{'G2x'});
conv3Block = dagnn.Conv('size',[1 1 500 784],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('G_conv3',conv3Block,{'G2x'},{'G3'},{'G3f','G3b'});
net.addLayer('Grelu3',SigmoidBlock,{'G3'},{'G3x'});

%data
net.addLayer('concat',dagnn.Concat('dim',4),{'G3x','data_gt'},{'data'});
net.addLayer('dropout_data',dagnn.GausianNoise('rate',0.2),{'data'},{'data_d'});

%D
conv4Block = dagnn.Conv('size',[1 1 784 1000],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv1',conv4Block,{'data_d'},{'D1'},{'D1f','D1b'});
%net.addLayer('D_bn1',dagnn.BatchNorm(),{'D1'},{'D1bn'},{'Dbn1f','Dbn1b','Dbn1c'});
net.addLayer('Drelu1',reluBlock,{'D1'},{'D1x'});
%net.addLayer('Ddropout1',dagnn.GausianNoise('rate',0.5),{'D1x'},{'D1d'});

conv5Block = dagnn.Conv('size',[1 1 1000 500],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv2',conv5Block,{'D1x'},{'D2'},{'D2f','D2b'});
%net.addLayer('D_bn2',dagnn.BatchNorm(),{'D2'},{'D2bn'},{'Dbn2f','Dbn2b','Dbn2c'});
net.addLayer('Drelu2',reluBlock,{'D2'},{'D2x'});
net.addLayer('Ddropout2',dagnn.GausianNoise('rate',0.5),{'D2x'},{'D2d'});

conv6Block = dagnn.Conv('size',[1 1 500 250],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv3',conv6Block,{'D2d'},{'D3'},{'D3f','D3b'});
%net.addLayer('D_bn3',dagnn.BatchNorm(),{'D3'},{'D3bn'},{'Dbn3f','Dbn3b','Dbn3c'});
net.addLayer('Drelu3',reluBlock,{'D3'},{'D3x'});
net.addLayer('Ddropout3',dagnn.GausianNoise('rate',0.5),{'D3x'},{'D3d'});

conv7Block = dagnn.Conv('size',[1 1 250 250],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv4',conv7Block,{'D3d'},{'D4'},{'D4f','D4b'});
%net.addLayer('D_bn4',dagnn.BatchNorm(),{'D4'},{'D4bn'},{'Dbn4f','Dbn4b','Dbn4c'});
net.addLayer('Drelu4',reluBlock,{'D4'},{'D4x'});
net.addLayer('Ddropout4',dagnn.GausianNoise('rate',0.5),{'D4x'},{'D4d'});

conv8Block = dagnn.Conv('size',[1 1 250 250],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]); %4*4
net.addLayer('Dconv5',conv8Block,{'D4d'},{'D5'},{'D5f','D5b'});
%net.addLayer('D_bn5',dagnn.BatchNorm(),{'D5'},{'D5bn'},{'Dbn5f','Dbn5b','Dbn5c'});
net.addLayer('Drelu5',reluBlock,{'D5'},{'D5x'});
net.addLayer('Ddropout5',dagnn.GausianNoise('rate',0.5),{'D5x'},{'D5d'});

%truth or fake loss
fcBlock = dagnn.Conv('size',[1 1 250 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('Dfc',fcBlock,{'D5d'},{'prediction'},{'Dfcf','Dfcb'});
lossBlock = dagnn.Loss('loss', 'softmaxlog');
net.addLayer('Dloss',lossBlock,{'prediction','label'},'Dobjective');
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;

%classify loss
fc2Block = dagnn.Conv('size',[1 1 250 10],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('GDfc2',fc2Block,{'D5d'},{'prediction2'},{'GDfcf','GDfcb'});
net.addLayer('GDloss',lossBlock,{'prediction2','label2'},'GDobjective');
net.addLayer('top1err2', dagnn.Loss('loss', 'classerror'), ...
    {'prediction2','label2'}, 'top1err2') ;

%feature matching
net.addLayer('Gloss',dagnn.Feature_Match_Loss(),{'D1x','prediction'},'Gobjective');

net.initParams();
end

