function ok = checkProvenance(tbl)
% checkProvenance  Refuse to let a table mix code versions.
%
% The ledger holds many sweeps; the runs behind one table must come from one.
% The checks run per sweep.
if nargin<1
    root = fullfile(fileparts(mfilename('fullpath')),'..','..');
    tbl = readtable(fullfile(root,'results','runs.csv'));
end
ok = true;
sweeps = unique(tbl.sweep);
if numel(sweeps)>1
    fprintf('checkProvenance: ledger holds %d sweeps (%s); checking each separately.\n', ...
            numel(sweeps), strjoin(sweeps',', '));
end
for si = 1:numel(sweeps)
    sub = tbl(strcmp(tbl.sweep,sweeps{si}),:);
    cm  = unique(sub.commit);
    if numel(cm)>1
        % Two commits split a sweep only if they differ in a path that can change
        % a number. The paths are listed in results/computational-paths.txt.
        d = computationalDiff(cm{1},cm(2:end));
        if isempty(d)
            fprintf(['checkProvenance: sweep %s spans %d commits (%s), identical in ' ...
                     'every computational path; treated as one experiment.\n'], ...
                    sweeps{si}, numel(cm), strjoin(cm',', '));
        else
            ok = false;
            warning('checkProvenance:mixedCommits', ...
                ['Sweep %s spans %d commits: %s. They differ in %s, so its runs are ' ...
                 'not one experiment.'], ...
                sweeps{si}, numel(cm), strjoin(cm',', '), strjoin(d,', '));
        end
    end
end
if any(strcmp(tbl.commit,'unknown'))
    ok = false;
    warning('checkProvenance:unknownCommit', ...
        '%d run(s) have no commit recorded.', sum(strcmp(tbl.commit,'unknown')));
end
% Ragged seed coverage within a sweep. The key includes alpha, so a sweep that
% varies alpha is not mistaken for uneven seeds.
key = strcat(tbl.dataset,'|',tbl.dynamic,'|',tbl.basis,'|',string(tbl.d), ...
             '|',string(tbl.T),'|',tbl.opti,'|',string(tbl.alpha1),'|',string(tbl.alpha2), ...
             '|',string(tbl.precision),'|',string(tbl.maxEpochs),'|',string(tbl.warmStartW));
root = fullfile(fileparts(mfilename('fullpath')),'..','..');
for si = 1:numel(sweeps)
    m = strcmp(tbl.sweep,sweeps{si});
    [u,~,ix] = unique(key(m)); n = accumarray(ix,1);
    if numel(unique(n))>1
        % A top-up sweep adds seeds to cells whose first seeds live in an earlier
        % sweep. Its manifest declares this; an undeclared ragged sweep is an error.
        if isTopUp(root,sweeps{si})
            fprintf(['checkProvenance: sweep %s contributes %d..%d runs per cell, ' ...
                     'declared a top-up of earlier sweeps.\n'], sweeps{si}, min(n), max(n));
        else
            ok = false;
            warning('checkProvenance:raggedSeeds', ...
                'Sweep %s has uneven seed counts (%d..%d). Example short config: %s', ...
                sweeps{si}, min(n), max(n), u{find(n==min(n),1)});
        end
    end
end
if ok, fprintf('checkProvenance: every sweep is one code state with even seed coverage.\n'); end
end

function paths = computationalPaths(sections)
% Read results/computational-paths.txt: the paths that can change a number, for the
% named pipelines.
root  = fullfile(fileparts(mfilename('fullpath')),'..','..');
lines = strtrim(strsplit(fileread(fullfile(root,'results','computational-paths.txt')),newline));
paths = {}; cur = '';
for i = 1:numel(lines)
    L = lines{i};
    if isempty(L) || startsWith(L,'#'), continue, end
    if startsWith(L,'[') && endsWith(L,']')
        cur = L(2:end-1);
    elseif any(strcmp(cur,sections))
        paths{end+1} = L; %#ok<AGROW>
    end
end
assert(~isempty(paths),'checkProvenance:noPaths', ...
    'computational-paths.txt lists nothing for %s', strjoin(sections,', '));
end

function d = computationalDiff(base,others)
% Files differing between base and each of others, restricted to computational paths.
paths = computationalPaths({'matlab','shared'});
d = {};
for i = 1:numel(others)
    % --no-pager: a pager would inject escape codes into the captured output.
    cmd = sprintf('git --no-pager diff --name-only %s %s -- %s', base, others{i}, strjoin(paths,' '));
    [st,out] = system(cmd);
    if st~=0
        % A commit that is not in this clone cannot be shown to be equivalent.
        d = [d, {sprintf('<could not diff %s..%s>',base,others{i})}]; %#ok<AGROW>
        continue
    end
    out = strtrim(strsplit(strtrim(out),newline));
    d = [d, out(~cellfun(@isempty,out))]; %#ok<AGROW>
end
d = unique(d);
end

function tf = isTopUp(root,sweepId)
% Does this sweep's manifest declare it a top-up of earlier sweeps?
tf = false;
f = fullfile(root,'results','sweeps',[sweepId '.json']);
if ~exist(f,'file'), return, end
m = jsondecode(fileread(f));
if ~isfield(m,'topUp'), return, end
assert(isfield(m.topUp,'topsUp') && isfield(m.topUp,'reason'), ...
    'checkProvenance:badTopUp', ...
    ['Sweep %s declares "topUp" without topsUp and reason. Excusing ragged seed ' ...
     'coverage must say which sweeps are extended, and why.'], sweepId);
tf = true;
end
