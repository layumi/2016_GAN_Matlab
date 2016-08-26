%im_retrieval
opts.dataDir = './scale_minist';  %define where to save data
opts.outputsize = 256;
files = {'train-images-idx3-ubyte', ...
         'train-labels-idx1-ubyte', ...
         't10k-images-idx3-ubyte', ...
         't10k-labels-idx1-ubyte'} ;

if ~exist(opts.dataDir, 'dir')
  mkdir(opts.dataDir) ;
end

for i=1:4
  if ~exist(fullfile(opts.dataDir, files{i}), 'file')
    url = sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',files{i}) ;
    fprintf('downloading %s\n', url) ;
    gunzip(url, opts.dataDir) ;
  end
end

f=fopen(fullfile(opts.dataDir, 'train-images-idx3-ubyte'),'r') ;
data1=fread(f,inf,'uint8');
fclose(f) ;
data1=permute(reshape(data1(17:end),28,28,60e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 't10k-images-idx3-ubyte'),'r') ;
data2=fread(f,inf,'uint8');
fclose(f) ;
data2=permute(reshape(data2(17:end),28,28,10e3),[2 1 3]) ;

f=fopen(fullfile(opts.dataDir, 'train-labels-idx1-ubyte'),'r') ;
label1=fread(f,inf,'uint8');
fclose(f) ;
label1=double(label1(9:end)')+1 ;

f=fopen(fullfile(opts.dataDir, 't10k-labels-idx1-ubyte'),'r') ;
label2=fread(f,inf,'uint8');
fclose(f) ;
label2=double(label2(9:end)')+1 ;

% 128*128
data = cat(3,data1,data2);
label = cat(2,label1,label2);
num = size(data,3);
sz = opts.outputsize;
DATA = zeros(sz,sz,num);
for i = 1:num
    factor = 8*rand()+1;
    im = imresize(data(:,:,i),factor,'bilinear');
    [h,w,~] = size(im);
    x1 = randi(sz-h+1);
    y1 = randi(sz-w+1);
    DATA(x1:x1+h-1,y1:y1+w-1,i) = im;
end
imdb.images.set = [ones(1,60000),3*ones(1,10000)];
DATA = reshape(DATA,sz,sz,1,[]);
dataMean = mean(DATA(:,:,:,imdb.images.set == 1), 4);
imdb.images.data = DATA;
imdb.images.label = label;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.images.data_mean = dataMean;

index = find(imdb.images.label==9);
hold off;
for i=1:10
    subplot(2,5,i);hold on;
    imshow(imdb.images.data(:,:,:,index(i)));
end
save('scale_minist.mat','imdb','-v7.3');