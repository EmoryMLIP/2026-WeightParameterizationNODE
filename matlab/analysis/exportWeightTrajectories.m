function exportWeightTrajectories(sweepId,seeds,outMat)
% exportWeightTrajectories  Learned dynamics weights over time, per step.
%
% For ELM, CDR and DCR, the ResNet at T=1 under ADAM: the degree-3 Legendre model
% and the unconstrained model (one weight set per step).
%
% Writes, per model, the weights of the dynamics layer at the nt time points,
% using the network's own split and basis matrix, so the values are those of the
% forward pass. Needs results/raw/, which is not shipped. The figure is drawn
% from the output by python/export_weight_figure.py.
if nargin<1 || isempty(sweepId), sweepId = '2026-10-02-paper'; end
if nargin<2 || isempty(seeds), seeds = 20:29; end
if nargin<3 || isempty(outMat)
    outMat = fullfile(fileparts(mfilename('fullpath')),'..','..','results','weight_trajectories.mat');
end
addpath(genpath(fullfile(fileparts(mfilename('fullpath')),'..')));
p = sweepPaths(sweepId);

models = {'ELM','Legendre',3; 'ELM','none',0; 'CDR','Legendre',3; 'CDR','none',0; ...
          'DCR','Legendre',3; 'DCR','none',0};
out = struct('dataset',{},'basis',{},'d',{},'seed',{},'t',{},'K',{},'b',{},'testErr',{});
for seed = seeds
for i = 1:size(models,1)
    [ds,basis,d] = deal(models{i,:});
    tag = runTag(ds,'ResNN',basis,d,1,'sgd',seed);
    S = load(fullfile(p.raw,[tag '.mat']));
    r = S.results;
    th = split(r.net,r.thOpt);
    blk = r.net.blocks{2};                          % the ResNN block
    nl = nTheta(blk.layer);
    Theta = reshape(th{2},nl,[]) * blk.A;           % nl x nt, as in ResNN.forwardProp
    nc = sqrt(nl + 1/4) - 1/2;                      % nl = nc^2 + nc (dense K plus bias)
    assert(abs(nc-round(nc))<1e-12,'unexpected layer size %d',nl); nc = round(nc);
    nt = size(Theta,2);
    j = numel(out) + 1;
    out(j).dataset = ds; out(j).basis = basis; out(j).d = d; out(j).seed = seed;
    out(j).t = linspace(0,1,nt);                    % the weight nodes of getPolynomialBasis
    out(j).K = Theta(1:nc^2,:);
    out(j).b = Theta(nc^2+1:end,:);
    out(j).testErr = r.err.test.aggregate;
end
end
prov = provenanceLine('');
save(outMat,'out','sweepId','seeds','prov','-v7');
fprintf('exportWeightTrajectories: wrote %s\n',outMat);
end
