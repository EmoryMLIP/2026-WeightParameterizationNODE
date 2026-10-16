function runTests()
% runTests  Run every MATLAB test in this folder. Errors on the first failure.
tests = {@testBasis, @testVarProGram, @testSGD};
for k = 1:numel(tests)
    fprintf('\n########## %s ##########\n', func2str(tests{k}));
    tests{k}();
end
fprintf('\n===== all MATLAB tests passed =====\n');
