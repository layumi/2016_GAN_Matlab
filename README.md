# Generative Adversarial Nets for Matlab
!{}(https://github.com/layumi/2016_GAN_Matlab/blob/master/show.png)

I use feature matching to train Generative model. (I define this Loss in the `/matlab/+dagnn/Feature_Match_Loss.m`)

1.You can test this code by run `test_gan_3.m`  

2.If you wanna train this code, you can run `train_gan_3.m`
You can find the network structure `GDnet_3.m`


# Some Details
1.I may miss some thing or not select a good initial parameter. So any advice is welcome. 

2.
GDnet_1 is using 32*32 random map as input

GDnet_2 is using 100 random vector and using deconv 

GDnet_3 is using 100 random vector and using conv (like fc layer)

In my experiment, deconv show that the output adjacent pixel is likely.
So in the minist using conv(fc layer) is better. (deconv may suit for real images such as CIFAR)   
