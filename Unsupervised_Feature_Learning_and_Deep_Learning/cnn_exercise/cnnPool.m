function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

for featureNum = 1:numFeatures
  for imageNum = 1:numImages
    for poolRow = floor(convolvedDim / poolDim)
      offsetRow = 1 + (poolRow-1) * poolDim;
      for poolCol = 1:floor(convolvedDim / poolDim)
		offsetCol = 1 + (poolCol-1) * poolDim;
		patch = convolvedFeatures(featureNum, imageNum, ...
				  offsetRow:offsetRow + poolDim-1, ...
				  offsetCol:offsetCol+poolDim-1);
		pooledFeatures(featureNum, imageNum, poolRow, poolCol) = mean(patch(:));
      end
    end
  end
end

end

