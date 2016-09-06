classdef Feature_Match_Loss < dagnn.Loss

  methods
    function outputs = forward(obj, inputs, params)
      [w,h,c,batchsize] = size(inputs{1});
      % inputs1 = reshape(inputs{1},[],batchsize);
      half = floor(batchsize/2);
      ra = inputs{1}(:,:,:,1:half);
      gt = inputs{1}(:,:,:,half+1:2*half);
      loss =  (gt-ra).^2;
      prediction = inputs{2};
      [~,index] = max(prediction,[],3);
      wrong =  index==1;
      wrong = repmat(wrong(:,:,:,1:half),[w,h,c,1]);
      outputs{1} = sum(sum(sum(sum(wrong.*loss))))/100;
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [w,h,c,batchsize] = size(inputs{1});
      half = floor(batchsize/2);
      ra = inputs{1}(:,:,:,1:half);
      gt = inputs{1}(:,:,:,half+1:2*half);
      prediction = inputs{2};
      [~,index] = max(prediction,[],3);
      wrong =  index==1;
      wrong = repmat(wrong(:,:,:,1:half),[w,h,c,1]);
      dra = gather(2.*(ra-gt) .* derOutputs{1}).*wrong;
      dra(find(dra>1)) = 1;
      dra(find(dra<-1)) = -1;
      dgt = zeros(w,h,c,batchsize-half,'single');
      derInputs{1} = gpuArray(cat(4,dra,dgt));
      derInputs{2} = [];
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Feature_Match_Loss(varargin)
      obj.load(varargin) ;
    end
  end
end
