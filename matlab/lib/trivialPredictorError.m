function s = trivialPredictorError(Ctrain,C)
% trivialPredictorError  Relative errors of predicting the training mean.
s = relativeErrors(repmat(mean(Ctrain,2),1,size(C,2)), C);
