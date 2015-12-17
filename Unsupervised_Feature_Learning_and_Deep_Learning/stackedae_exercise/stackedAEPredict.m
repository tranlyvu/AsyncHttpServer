function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

W1 = stack{1}.w;
b1 = stack{1}.b;
activation2 = sigmoid(W1 * data + repmat(b1, 1, inputSize));

% Second hidden layer activiations
W2 = stack{2}.w;
b2 = stack{2}.b;
activation3 = sigmoid(W2 * activation2 + repmat(b2, 1, size(activation2, 2)));

% Softmax predictions
M = softmaxTheta * activation3;
M = bsxfun(@minus, M, max(M, [], 1));
M = exp(M);
M = bsxfun(@rdivide, M, sum(M));
[~, pred] = max(M, [], 1);










% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
