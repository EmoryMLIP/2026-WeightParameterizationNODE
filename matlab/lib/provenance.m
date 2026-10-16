function p = provenance()
% provenance  Stamp identifying the exact code state that produced an artifact.
% Returns a struct; provenanceLine formats it as a one-line comment.
p = struct();
try
    [st,out] = system('git rev-parse --short HEAD');
    p.commit = strtrim(out); if st~=0, p.commit = 'unknown'; end
    % Dirty means uncommitted code. Generated artifacts (tables/, figures/,
    % results/) are excluded because writing one dirties the tree.
    [~,dirty] = system('git status --porcelain -- code data Makefile');
    p.dirty = ~isempty(strtrim(dirty));
catch
    p.commit = 'unknown'; p.dirty = true;
end
p.when    = datestr(now,'yyyy-mm-dd HH:MM:SS');
p.matlab  = version('-release');
% getenv rather than java.net.InetAddress: the latter throws under -nojvm.
p.host    = getenv('HOSTNAME');
if isempty(p.host), p.host = getenv('COMPUTERNAME'); end
if isempty(p.host), [st,h] = system('hostname'); if st==0, p.host = strtrim(h); end, end
if isempty(p.host), p.host = 'unknown'; end
