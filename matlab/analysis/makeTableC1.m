function makeTableC1(sweepId,runsCsv,outTex)
% makeTableC1  Emit tables/tableC1.tex directly from the run ledger.
%
% Reads results/runs.csv (written by collectResults.m). Emits only the tabular
% body; the caption and label live in the manuscript.

if nargin<1, sweepId = ''; end
if nargin<2 || isempty(runsCsv)
    runsCsv = fullfile(fileparts(mfilename('fullpath')),'..','..','results','runs.csv');
end
if nargin<3 || isempty(outTex)
    outTex = fullfile(fileparts(mfilename('fullpath')),'..','..','tables','tableC1.tex');
end
T = readtable(runsCsv);
% One table must come from one sweep.
if isempty(sweepId)
    % Not the ledger's last sweep: the GNvpro seed top-up sorts after the paper
    % sweep and holds no monomial runs.
    [~,sweepId] = selectDefaultRuns(T);
end
T = T(strcmp(T.sweep,sweepId),:);
if isempty(T), error('makeTableC1:noRuns','no runs for sweep %s in %s',sweepId,runsCsv); end

% Drop the readout warm-start ablation (differs only in warmStartW).
if ismember('warmStartW',T.Properties.VariableNames)
    keep = true(height(T),1);
    for ds = unique(T.dataset)'
        cfg = getDatasetConfig(ds{1});
        m = strcmp(T.dataset,ds{1});
        keep(m) = isnan(T.warmStartW(m)) | T.warmStartW(m) == cfg.warmStartW;
    end
    T = T(keep,:);
end
% Drop the regularization and single-precision checks.
T = T(T.alpha1 == 1e-10 & strcmp(T.precision,'double'),:);

datasets = {'ELM','DCR','CDR'};

fid = fopen(outTex,'w');
c = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',provenanceLine('%'));
[~,csvName,csvExt] = fileparts(runsCsv);
fprintf(fid,'%% source: results/%s%s   sweep: %s\n',csvName,csvExt,sweepId);
fprintf(fid,'%% regenerate with: makeTableC1(''%s'')\n',sweepId);
fprintf(fid,'%% metric: aggregate relative error ||Pred-C||_F/||C||_F on the test set\n');
fprintf(fid,'%% cells are the geometric mean over %d seeds (geometric standard deviation as a factor),\n',numel(unique(T.seed)));
fprintf(fid,'%% printed as a two-digit mantissa against the power of ten named with each dataset\n');
fprintf(fid,'\\begin{tabular}{llcccccc}\n\\toprule\n');
fprintf(fid,'& & & & \\multicolumn{2}{c}{ResNet} & \\multicolumn{2}{c}{Hamiltonian} \\\\\n');
fprintf(fid,'\\cmidrule(lr){5-6} \\cmidrule(lr){7-8}\n');
fprintf(fid,'Dataset & Parameterization & Params & $T$ & Adam & GNvpro & Adam & GNvpro \\\\\n\\midrule\n');

for di = 1:numel(datasets)
    ds = datasets{di};
    % One power of ten per dataset, as in tables/table3.tex: the datasets differ
    % by two orders of magnitude.
    r = T(strcmp(T.dataset,ds),:);
    if isempty(r), continue, end
    e = floor(log10(exp(mean(log(r.test_agg)))));
    emitBlock(fid,T,ds,'Legendre',3,[1 5 10],'Legendre ($D=3$)',e,ds, ...
              sprintf('$(\\times 10^{%d})$',e));
    emitBlock(fid,T,ds,'monomial',3,[1 5 10],'monomial ($D=3$)',e);
    % Degrees 4-6 appear in the degree-axis figure (figures/degree_lever.tex).
    emitBlock(fid,T,ds,'none',0,[1 5 10],'none',e);
    % trivial-predictor reference, so every value has a baseline in view
    fprintf(fid,'\\cmidrule(lr){2-8}\n');
    fprintf(fid,'& \\emph{mean predictor} & & & \\multicolumn{4}{c}{%s} \\\\\n', ...
        mant(mean(r.trivial_test_agg)/10^e));
    if di<numel(datasets), fprintf(fid,'\\midrule\n'); end
end
fprintf(fid,'\\bottomrule\n\\end{tabular}\n');
fprintf('makeTableC1: wrote %s\n',outTex);
end

function emitBlock(fid,T,ds,basis,deg,Ts,label,e,dsLabel,dsLabel2)
% dsLabel fills the Dataset column on this block's first row.
if nargin < 9, dsLabel = ''; end
if nargin < 10, dsLabel2 = ''; end   % second row: the power of ten
rows = T(strcmp(T.dataset,ds) & strcmp(T.basis,basis) & T.d==deg,:);
if isempty(rows), return; end
fprintf(fid,'\\cmidrule(lr){2-8}\n');
for k = 1:numel(Ts)
    Tval = Ts(k);
    sel = rows(rows.T==Tval,:);
    if isempty(sel), continue; end
    vals = nan(1,4); sds = nan(1,4); ns = zeros(1,4);
    specs = {'ResNN','sgd'; 'ResNN','GNvpro'; 'hamiltonian','sgd'; 'hamiltonian','GNvpro'};
    for j = 1:4
        m = sel(strcmp(sel.dynamic,specs{j,1}) & strcmp(sel.opti,specs{j,2}),:);
        % Geometric mean and standard deviation: the paper's comparisons are paired
        % ratios on log test error.
        if ~isempty(m), vals(j) = exp(mean(log(m.test_agg))); sds(j) = exp(std(log(m.test_agg))); ns(j) = height(m); end
    end
    if k==1
        n = sel.nParamsDynamics(1);
        pre = sprintf('%s & \\multirow{%d}{*}{%s} & \\multirow{%d}{*}{%d}', ...
                      dsLabel,numel(Ts),label,numel(Ts),n);
        dsLabel = '';
    else
        pre = sprintf('%s & & ',dsLabel2);
        dsLabel2 = '';
    end
    cells = arrayfun(@(v,sd) fmt(v/10^e,sd), vals, sds, 'uni', 0);
    fprintf(fid,'%s & $T=%g$ & %s \\\\  %% n=%s seeds\n',pre,Tval,strjoin(cells,' & '),mat2str(ns));
end
end

function s = fmt(v,sd)
% Geometric mean over seeds with the geometric standard deviation as a
% multiplicative factor.
if isnan(v)
    s = '';   % an empty cell, not an em dash
elseif nargin<2 || isnan(sd) || sd==1
    s = mant(v);
else
    s = sprintf('$%s\\,(%.2f)$',mant(v),sd);
end
end

function s = mant(m)
% Two significant digits for mantissas in [1,100); %.2g would print 100 as 1e+02.
if m >= 10
    s = sprintf('%.0f',m);
else
    s = sprintf('%.1f',m);
end
end
