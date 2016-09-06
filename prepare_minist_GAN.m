%im_retrieval
opts.dataDir = './minist_data';  %define where to save data
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

data = cat(3,data1,data2);
data = reshape(data,28,28,1,[]);
label = cat(2,label1,label2);

imdb.images.set = [ones(1,60000),2*ones(1,10000)];
dataMean = mean(data(:,:,:,imdb.images.set == 1), 4);
imdb.images.data = data;
imdb.images.label = label;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.images.data_mean = dataMean;
save('minist_data.mat','imdb','-v7.3');

%only 2
imdb.images.data = reshape(data(:,:,find(imdb.images.label ==3)),28,28,1,[]);
imdb.images.set = ones(1,numel(find(imdb.images.label ==3)));
imdb.images.label = ones(1,numel(find(imdb.images.label ==3)));
save('minist_data_only2.mat','imdb','-v7.3');