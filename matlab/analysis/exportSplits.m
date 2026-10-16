function exportSplits(outFile)
% exportSplits  Write the CDR/DCR train/val/test index permutations to disk so
% the PyTorch pipeline uses the same split as MATLAB.
%
% MATLAB's randperm and NumPy's permutation order differently under the same
% seed, so the indices are shared directly. Run after changing getDatasetConfig.m:
%
%   >> exportSplits
%
% ELM is not included: its split is baked into NNERDS.mat and both pipelines
% already read it directly.
if nargin<1 || isempty(outFile)
    outFile = fullfile(fileparts(mfilename('fullpath')),'..','..','data','splits.mat');
end
splits = struct();
for ds = {'CDR','DCR'}
    name = ds{1};
    cfg  = getDatasetConfig(name);
    rng(42);                       % same seed the experiment driver uses
    switch name
        case 'CDR', S = load('CDR_Data.mat');
        case 'DCR', S = load('DCR_Data.mat');
    end
    n   = size(S.Y,2);
    idx = randperm(n);             % consumed exactly as setup{CDR,DCR}.m does
    splits.(name) = struct('idx',int32(idx),'nTrain',cfg.nTrain,'nVal',cfg.nVal,'n',n);
    fprintf('%s: n=%d nTrain=%d nVal=%d nTest=%d\n',name,n,cfg.nTrain,cfg.nVal,n-cfg.nTrain-cfg.nVal);
end
save(outFile,'-struct','splits');
fprintf('wrote %s\n',outFile);
