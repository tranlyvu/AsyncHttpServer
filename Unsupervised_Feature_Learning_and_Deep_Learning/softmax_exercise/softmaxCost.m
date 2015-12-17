function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;

thetagrad = zeros(numClasses, inputSize);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

h=theta*data;
h=bsxfun(@minus, h, max(h, [], 1));
h=exp(h);
h = h ./ repmat(sum(h, 1), numClasses, 1);

%calculate cost
cost=cost+sum(sum(groundTruth*log(h)',1));
cost=cost*(-1/numCases);
weight_decay=sum(sum(theta.^2));
weight_decay=weight_decay*lambda*0.5;
cost=cost+weight_decay;

%calculate thetagrad
thetagrad=(groundTruth-log(h))*data';
thetagrad=thetagrad./numCases;
thetagrad=thetagrad + lambda*theta;






% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

