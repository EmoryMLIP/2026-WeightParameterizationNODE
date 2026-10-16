function testVarProGram()
% Verifies the Gram-matrix regularizer inside dnnVarProRegressionObjFctn:
% (a) Btheta = [] matches Btheta = I, (b) the objective value equals
% the closed form, (c) the derivative is still correct (Meganet checkDerivative).
addpath(genpath('..'));
np=0; nf=0;
function chk(name,cond,detail)
    if cond, np=np+1; fprintf('  PASS  %s\n',name);
    else,    nf=nf+1; fprintf(2,'  FAIL  %s   %s\n',name,detail); end
end
maxNumCompThreads(1); rng(7);

nc=5; nt=8; T=2.5; d=3; alpha1=0.7; alpha2=1e-3;
Y=randn(4,60); C=randn(3,60);
block1=NN({singleLayer(dense([nc,4],'Bin',eye(nc)))});
[A,M]=getPolynomialBasis(d,T,nt,'monomial');       % dense Gram -> nontrivial B
block2=ResNN(singleLayer(dense([nc,nc]),'Bin',eye(nc)),nt,T/nt,'A',A');
net=Meganet({block1,block2}); pLoss=regressionLoss();
n1=nTheta(block1); nLayer=nTheta(block2)/size(A,2);
B=blkdiag(speye(n1),kron(sparse(chol(M)),speye(nLayer)));

f0=dnnVarProRegressionObjFctn(net,pLoss,Y,C,'alpha1',alpha1,'alpha2',alpha2);
fI=dnnVarProRegressionObjFctn(net,pLoss,Y,C,'alpha1',alpha1,'alpha2',alpha2,'Btheta',speye(nTheta(net)));
fB=dnnVarProRegressionObjFctn(net,pLoss,Y,C,'alpha1',alpha1,'alpha2',alpha2,'Btheta',B);

th=initTheta(net); th=th+0.1*randn(size(th));
fprintf('\n== 1. Btheta = [] and Btheta = I agree (default regularizer unchanged) ==\n');
[J0,~,dJ0]=eval(f0,th); [JI,~,dJI]=eval(fI,th);
chk('objective identical', abs(J0-JI)/abs(J0)<1e-12, sprintf('%.6g vs %.6g',J0,JI));
chk('gradient identical', norm(dJ0-dJI)/norm(dJ0)<1e-12, sprintf('%.2e',norm(dJ0-dJI)));

fprintf('\n== 2. Btheta = chol(M) kron I gives the Gram-weighted penalty ==\n');
[JB,paraB]=eval(fB,th);
% difference in objective must equal the difference in the theta-regularizer
Th=reshape(th(n1+1:end),nLayer,d+1);
Rgram = 0.5*alpha1*( th(1:n1)'*th(1:n1) + trace(Th*M*Th') );
Reucl = 0.5*alpha1*(th'*th);
chk('J_B - J_I == R_gram - R_euclid', abs((JB-JI)-(Rgram-Reucl))/abs(Rgram-Reucl)<1e-10, ...
    sprintf('%.8g vs %.8g',JB-JI,Rgram-Reucl));
chk('Gram penalty differs from Euclidean (test is meaningful)', ...
    abs(Rgram-Reucl)/abs(Reucl)>1e-3, 'M is too close to I');

fprintf('\n== 3. derivative check on the Gram-weighted objective ==\n');
for lbl_f = {{'Btheta=[]',f0},{'Btheta=chol(M)kronI',fB}}
    lbl=lbl_f{1}{1}; fc=lbl_f{1}{2};
    rng(11);
    [isOK,Err,Order] = checkDerivative(fc,th,'out',0);
    % Err(:,2) is the second-order Taylor residual |F(x+hv)-F-h<dF,v>|
    e = Err(:,2); e = e(e>1e-13);
    r = e(1:end-1)./e(2:end);
    chk(sprintf('%s: checkDerivative isOK',lbl), isOK, 'Taylor test failed');
    chk(sprintf('%s: 2nd-order Taylor rate ~4 (median %.2f)',lbl,median(r)), ...
        abs(median(r)-4)<0.8, sprintf('median %.3f',median(r)));
end

fprintf('\n---------------------------------------------\n');
fprintf('%d passed, %d failed\n',np,nf);
if nf>0, error('testVarProGram:failures','%d test(s) failed',nf); end
end
