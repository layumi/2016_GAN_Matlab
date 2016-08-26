addpath draw;
net = cnn_stn_cluttered_mnist_init([60 60], true) ;
draw_full_net(net,'stn');