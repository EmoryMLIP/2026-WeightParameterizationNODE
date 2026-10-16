function testSGD()
% Tests for the early-stopping and best-checkpoint options of sgd.m.
% Verifies (a) training is deterministic and runs the full budget when patience = Inf,
% (b) the best-validation iterate is returned, not the last one,
% (c) patience actually terminates training.
addpath(genpath('..'));
addpath(fullfile('..','..','data'));
np=0; nf=0;
function chk(name,cond,detail)
    if cond, np=np+1; fprintf('  PASS  %s\n',name);
    else,    nf=nf+1; fprintf(2,'  FAIL  %s   %s\n',name,detail); end
end

% small deterministic regression problem
maxNumCompThreads(1);
rng(42);
n=40; Yt=randn(4,n); Ct=[sum(Yt.^2,1); Yt(1,:)]; Yv=randn(4,20); Cv=[sum(Yv.^2,1); Yv(1,:)];
nc=6; nt=6; T=1;
block1=NN({singleLayer(dense([nc,4],'Bin',eye(nc)))});
A=getPolynomialBasis(2,T,nt,'Legendre');
block2=ResNN(singleLayer(dense([nc,nc]),'Bin',eye(nc)),nt,T/nt,'A',A');
net=Meganet({block1,block2}); pLoss=regressionLoss();

    function [xEnd,xBest,His,info] = run(patience,maxEp)
        rng(20);
        o=sgd('nesterov',false,'ADAM',true,'miniBatch',8,'out',0,'lossTol',-Inf);
        o.learningRate=1e-2; o.maxEpochs=maxEp; o.earlyStopPatience=patience;
        th0=initTheta(net);
        YN=forwardProp(net,th0,Yt); X=[YN;ones(1,size(YN,2))];
        W=(Ct*X')/(X*X'+1e-3*eye(size(X,1)));
        f=dnnObjFctn(net,tikhonovReg(opEye(numel(th0)),1e-5),pLoss, ...
                     tikhonovReg(opEye(numel(W)),1e-3),Yt,Ct);
        fv=dnnObjFctn(net,[],pLoss,[],Yv,Cv);
        [xEnd,His,~,info]=solve(o,f,[th0;W(:)],fv);
        xBest=info.xOptLoss;
    end

fprintf('\n== 1. patience = Inf runs the full budget deterministically ==\n');
[xA,~,HA] = run(Inf,60);
[xB,~,HB] = run(Inf,60);
chk('deterministic across repeated runs', norm(xA-xB)==0, sprintf('%.2e',norm(xA-xB)));
chk('ran the full 60 epochs (no early exit)', size(HA.his,1)==60, sprintf('%d epochs',size(HA.his,1)));

fprintf('\n== 2. best-validation iterate is tracked and differs from the last ==\n');
[xEnd,xBest,His,info] = run(Inf,60);
vcol = find(strcmp(His.str,'F'),1,'last');       % validation objective column
vhis = His.his(:,vcol);
[~,argmin] = min(vhis);
chk('infoOptLoss returned', ~isempty(info) && isfield(info,'xOptLoss'),'missing');
chk(sprintf('optEpoch (%d) == argmin of logged val loss (%d)',info.optEpoch,argmin), ...
    info.optEpoch==argmin, sprintf('%d vs %d',info.optEpoch,argmin));
chk('optValLoss == min of logged val loss', abs(info.optValLoss-min(vhis))<1e-12, ...
    sprintf('%.8g vs %.8g',info.optValLoss,min(vhis)));
fprintf('     final-epoch val loss %.6g   best val loss %.6g (epoch %d)\n', vhis(end), min(vhis), argmin);
if argmin < numel(vhis)
    chk('best iterate differs from final iterate', norm(xBest-xEnd)>0, 'identical');
else
    fprintf('     (val loss still improving at the last epoch -- skipping differ check)\n');
end

fprintf('\n== 3. patience terminates training early ==\n');
[~,~,H5] = run(3,400);
chk('stopped before maxEpochs', size(H5.his,1)<400, sprintf('%d epochs',size(H5.his,1)));
fprintf('     stopped after %d epochs (patience 3)\n', size(H5.his,1));

fprintf('\n== 4. early-stopped run agrees with the un-stopped run up to that epoch ==\n');
[~,~,Hlong] = run(Inf,400);
k = size(H5.his,1);
chk('histories agree on the shared prefix', ...
    norm(H5.his(1:k,2)-Hlong.his(1:k,2))/norm(Hlong.his(1:k,2)) < 1e-12, ...
    sprintf('%.2e',norm(H5.his(1:k,2)-Hlong.his(1:k,2))));

fprintf('\n---------------------------------------------\n');
fprintf('%d passed, %d failed\n', np, nf);
if nf>0, error('testSGD:failures','%d test(s) failed',nf); end
end
