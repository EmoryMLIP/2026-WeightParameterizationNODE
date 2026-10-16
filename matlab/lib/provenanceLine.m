function s = provenanceLine(prefix)
% provenanceLine  One-line provenance comment for generated files.
if nargin<1, prefix = '%'; end
p = provenance();
s = sprintf('%s generated %s from commit %s%s by MATLAB R%s on %s', ...
    prefix, p.when, p.commit, ternary(p.dirty,' (WORKING TREE DIRTY)',''), p.matlab, p.host);
end
function o = ternary(c,a,b), if c, o=a; else, o=b; end, end
