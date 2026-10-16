function [Ytrain,Ctrain,Yval,Cval, Ytest,Ctest] = setupNNERDS()
% loads NNERDS dataset, normalizes, and splits into training, validation, test data


load NNERDS.mat


Ytrain = xall(idtrain,:)';
Ytrain = Ytrain - mean(paramrange,2);
Ytrain = 2* Ytrain ./ (paramrange(:,2)-paramrange(:,1));

Ctrain = yall(idtrain,:)';
minC = min(Ctrain,[],2);
maxC = max(Ctrain-minC,[],2);
Ctrain = (Ctrain-minC)./maxC;

Yval = xall(idval,:)';
Yval = Yval - mean(paramrange,2);
Yval = 2* Yval ./ (paramrange(:,2)-paramrange(:,1));

Cval = yall(idval,:)';
Cval = (Cval-minC)./maxC;

Ytest = xall(idtest,:)';
Ytest = Ytest - mean(paramrange,2);
Ytest = 2* Ytest ./ (paramrange(:,2)-paramrange(:,1));

Ctest = yall(idtest,:)';
Ctest = (Ctest-minC)./maxC;




