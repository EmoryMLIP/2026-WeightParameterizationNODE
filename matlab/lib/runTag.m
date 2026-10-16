function tag = runTag(dataset,dynamic,basis,d,T,opti,seed)
% runTag  The one filename convention, shared with python/sweep_paths.py.
%   <dataset>_<arch>_<basis>_d<D>_T<T>_<opti>_s<seed>
% The seed is always present so that reruns under different seeds have distinct names.
tag = sprintf('%s_%s_%s_d%d_T%g_%s_s%d',dataset,dynamic,basis,d,T,opti,seed);
