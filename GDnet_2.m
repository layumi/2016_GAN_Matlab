function net = GDnet_2()
net = dagnn.DagNN();
GreluBlock = dagnn.ReLU('leak',0);
reluBlock = dagnn.ReLU('leak',0.2);
SigmoidBlock = dagnn.Sigmoid();
%G
conv1Block = dagnn.ConvTranspose('size',[5 5 256 100],'hasBias',true,'upsample',[1,1],'crop',[0,0,0,0]);
net.addLayer('G_conv1',conv1Block,{'data_rand'},{'G1'},{'G1f','G1b'});
net.addLayer('G_bn1',dagnn.BatchNorm(),{'G1'},{'G1bn'},{'Gbn1f','Gbn1b','Gbn1c'});
net.addLayer('Grelu1',GreluBlock,{'G1bn'},{'G1x'});
conv2Block = dagnn.ConvTranspose('size',[5 5 128 256],'hasBias',true,'upsample',[2,2],'crop',[0,0,0,0]);
net.addLayer('G_conv2',conv2Block,{'G1x'},{'G2'},{'G2f','G2b'});
net.addLayer('G_bn2',dagnn.BatchNorm(),{'G2'},{'G2bn'},{'Gbn2f','Gbn2b','Gbn2c'});
net.addLayer('Grelu2',GreluBlock,{'G2bn'},{'G2x'});
conv3Block = dagnn.ConvTranspose('size',[4 4 1 128],'hasBias',true,'upsample',[2,2],'crop',[0,0,0,0]);
net.addLayer('G_conv3',conv3Block,{'G2x'},{'G3'},{'G3f','G3b'});
net.addLayer('Grelu3',SigmoidBlock,{'G3'},{'G3x'});
%conv4Block = dagnn.ConvTranspose('size',[5 5 1 256],'hasBias',true,'upsample',[2,2],'crop',[0,0,0,0]);
%net.addLayer('G_conv4',conv4Block,{'G4x'},{'G4'},{'G4f','G4b'});
%net.addLayer('Grelu4',SigmoidBlock,{'G4'},{'G4x'});

%data
net.addLayer('concat',dagnn.Concat('dim',4),{'G3x','data_gt'},{'data'});
%net.addLayer('dropout_data',dagnn.DropOut('rate',0.2),{'data'},{'data_d'});

%D
conv4Block = dagnn.Conv('size',[4 4 1 20],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]); %13*13*20
net.addLayer('Dconv1',conv4Block,{'data'},{'D1'},{'D1f','D1b'});
%net.addLayer('D_bn1',dagnn.BatchNorm(),{'D1'},{'D1bn'},{'Dbn1f','Dbn1b','Dbn1c'});
net.addLayer('Drelu1',reluBlock,{'D1'},{'D1x'});
%net.addLayer('Ddropout1',dagnn.GausianNoise('rate',0.5),{'D1x'},{'D1d'});

conv5Block = dagnn.Conv('size',[5 5 20 50],'hasBias',true,'stride',[2,2],'pad',[0,0,0,0]); %5*5*50
net.addLayer('Dconv2',conv5Block,{'D1x'},{'D2'},{'D2f','D2b'});
%net.addLayer('D_bn2',dagnn.BatchNorm(),{'D2'},{'D2bn'},{'Dbn2f','Dbn2b','Dbn2c'});
net.addLayer('Drelu2',reluBlock,{'D2'},{'D2x'});
net.addLayer('Ddropout2',dagnn.GausianNoise('rate',0.5),{'D2x'},{'D2d'});

conv6Block = dagnn.Conv('size',[5 5 50 500],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]); %1*1
net.addLayer('Dconv3',conv6Block,{'D2d'},{'D3'},{'D3f','D3b'});
%net.addLayer('D_bn3',dagnn.BatchNorm(),{'D3'},{'D3bn'},{'Dbn3f','Dbn3b','Dbn3c'});
net.addLayer('Drelu3',reluBlock,{'D3'},{'D3x'});
net.addLayer('Ddropout3',dagnn.GausianNoise('rate',0.5),{'D3x'},{'D3d'});

fcBlock = dagnn.Conv('size',[1 1 500 2],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('Dfc',fcBlock,{'D3d'},{'prediction'},{'Dfcf','Dfcb'});
lossBlock = dagnn.Loss('loss', 'softmaxlog');
net.addLayer('Dloss',lossBlock,{'prediction','label'},'Dobjective');
net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label'}, 'top1err') ;
%net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
 %     'opts', {'topk', 5}), {'prediction', 'label'}, 'top5err') ;
%net.addLayer('predict_label',dagnn.SoftMax(),'prediction','predict_label');

%convfBlock = dagnn.Conv('size',[4 4 50 500],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]); %4*4
%net.addLayer('Gconv4',convfBlock,{'D2x'},{'G4'},{'G4f','G4b'});
%net.addLayer('Gnormal',dagnn.LRN('param',[1000,0,1,0.5]),{'D3x'},{'G4n'});
net.addLayer('Gloss',dagnn.Feature_Match_Loss(),{'data','prediction'},'Gobjective');

net.initParams();
end

