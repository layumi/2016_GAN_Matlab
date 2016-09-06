classdef GausianNoise < dagnn.ElementWise
  properties
    rate = 0.5
    frozen = false
  end

  properties (Transient)
    mask
  end

  methods
    function outputs = forward(obj, inputs, params)
      if strcmp(obj.net.mode, 'test')
        outputs = inputs ;
        return ;
      end
      outputs{1} = inputs{1} + obj.rate*randn(size(inputs{1}));
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        derInputs = derOutputs ;
        derParams = {} ;
    end

    % ---------------------------------------------------------------------
    function obj = GausianNoise(varargin)
      obj.load(varargin{:}) ;
    end

    function obj = reset(obj)
      reset@dagnn.ElementWise(obj) ;
      obj.frozen = false ;
    end
  end
end
