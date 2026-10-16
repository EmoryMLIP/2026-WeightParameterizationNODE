function writeManifest(path,m)
% writeManifest  Persist a sweep manifest as JSON (pretty-printed).
fid = fopen(path,'w');
if fid==-1, error('writeManifest:cannotOpen','cannot open %s',path); end
c = onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(m,'PrettyPrint',true));
