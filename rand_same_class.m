function index = rand_same_class( imdb,label)
     list = find(imdb.images.label == label);
     num = numel(list);
     i = randi(num);
     index = list(i);
     while imdb.images.set(index) ~= 1
         i = randi(num);
         index = list(i);
     end
end

