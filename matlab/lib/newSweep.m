function sweepId = newSweep(slug,note)
% newSweep  Start a sweep: mint an id and write its manifest.
%
%   sweepId = newSweep('adjoint-newman','first sweep after the OTD switch')
%
% The manifest records the code state once per sweep, so a table built from it
% can name the commit behind every number.
if nargin<2, note = ''; end
assert(~isempty(regexp(slug,'^[a-z0-9-]+$','once')), ...
    'newSweep:badSlug','slug must be lowercase letters, digits and hyphens');
sweepId = sprintf('%s-%s',datestr(now,'yyyy-mm-dd'),slug);
p = sweepPaths(sweepId);
prov = provenance();
if prov.dirty
    warning('newSweep:dirtyTree', ...
        ['Working tree is DIRTY. Results will not be attributable to a commit. ' ...
         'Commit first unless this is a throwaway sweep.']);
end
m = struct('sweepId',sweepId,'slug',slug,'note',note, ...
           'commit',prov.commit,'dirty',prov.dirty,'started',prov.when, ...
           'matlabVersion',prov.matlab,'host',prov.host, ...
           'status','running','nOK',0,'nFail',0,'finished','');
writeManifest(p.manifest,m);
if prov.dirty, flag = ' [DIRTY]'; else, flag = ''; end
fprintf('sweep %s started (commit %s%s)\n',sweepId,prov.commit,flag);
end
