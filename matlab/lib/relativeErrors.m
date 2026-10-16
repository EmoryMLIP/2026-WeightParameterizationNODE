function s = relativeErrors(Pred,C)
% relativeErrors  Reporting metrics for a prediction Pred against targets C.
%
%   s.aggregate : ||Pred-C||_F / ||C||_F          (the reported metric)
%   s.median    : median_i ||p_i-c_i|| / ||c_i||
%   s.meanRatio : mean_i   ||p_i-c_i|| / ||c_i||  (unbounded, dominated by
%                 samples with small ||c_i||)
%
% Predicting the training mean scores 1.0 exactly on mean-centred targets (DCR),
% so values above 1 are worse than that baseline. trivialPredictorError returns
% the baseline for every dataset.

perSample   = sqrt(sum((Pred-C).^2,1)) ./ sqrt(sum(C.^2,1));
s.aggregate = norm(Pred-C,'fro') / norm(C,'fro');
s.median    = median(perSample);
s.meanRatio = mean(perSample);
s.perSample = perSample;
