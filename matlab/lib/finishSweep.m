function finishSweep(sweepId,nOK,nFail)
% finishSweep  Mark a sweep complete and record its tally.
% A sweep left in status 'running' crashed or was killed; its results are partial.
p = sweepPaths(sweepId);
m = jsondecode(fileread(p.manifest));
m.nOK = nOK; m.nFail = nFail;
m.finished = datestr(now,'yyyy-mm-dd HH:MM:SS');
if nFail==0, m.status = 'complete'; else, m.status = 'incomplete'; end
writeManifest(p.manifest,m);
fprintf('sweep %s %s: %d ok, %d failed\n',sweepId,m.status,nOK,nFail);
