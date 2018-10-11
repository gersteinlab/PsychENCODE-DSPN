% Code adapted from:
% Salakhutdinov, R. and Hinton, G. Deep Boltzmann Machines. AISTATS, 2009.

test_err=[];
test_crerr=[];
train_err=[];
train_crerr=[];

[numcases numdims numbatches]=size(batchdata);
N=numcases; 

[numdims numhids] = size(vishid);
[numhids numpens] = size(hidpen); 

%%%%%% Preprocess the data %%%%%%%%%%%%%%%%%%%%%%

[testnumcases testnumdims testnumbatches]=size(testdata);
N=testnumcases;
temp_h2_test = zeros(testnumcases,numpens,testnumbatches); 
for batch = 1:testnumbatches
   data = [testdata(:,:,batch)];
   [temp_h1, temp_h2] = ...
       dspn_mf_class(data,vishid,hidbiases,visbiases,hidpen,penbiases);
   temp_h2_test(:,:,batch) = temp_h2;
end  

[numcases numdims numbatches]=size(batchdata);
N=numcases;
temp_h2_train = zeros(numcases,numpens,numbatches);
for batch = 1:numbatches
   data = [batchdata(:,:,batch)];
   [temp_h1, temp_h2] = ...
     dspn_mf_class(data,vishid,hidbiases,visbiases,hidpen,penbiases);
   temp_h2_train(:,:,batch) = temp_h2;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

w1_penhid = hidpen';
w1_vishid = vishid;
w2 = hidpen;
h1_biases = hidbiases; h2_biases = penbiases; 
numlab = size(testdata_trait,2);

w_class = 0.1*randn(numpens,numlab); 
topbiases = 0.1*randn(1,numlab);

preds = [];

w1_vishid_all = [];
w2_all = [];
w_class_all = [];

for epoch = 1:maxepoch 

%%%% TEST STATS 
%%%% Error rates 
   [testnumcases testnumdims testnumbatches]=size(testdata);
   N=testnumcases;
   bias_hid = repmat(h1_biases,N,1);
   bias_pen = repmat(h2_biases,N,1);
   bias_top = repmat(topbiases,N,1);

   err=0;
   err_cr=0;
   counter=0;  
   totCount = 0;
   predVec = [];
   y_test = [];
   for batch = 1:testnumbatches
     data = [testdata(:,:,batch)];
     temp_h2 = temp_h2_test(:,:,batch); 
     target = [testdata_trait(:,:,batch)]; 

     w1probs = 1./(1 + exp(-data*w1_vishid -temp_h2*w1_penhid - bias_hid  )); 
     w2probs = 1./(1 + exp(-w1probs*w2 - bias_pen)); 
     targetout = exp(w2probs*w_class + bias_top );
     targetout = targetout./repmat(sum(targetout,2),1,numlab);
     [I J]=max(targetout,[],2); 
     [I1 J1]=max(target,[],2); 
     predVec = [predVec ; J];
     y_test = [y_test ; J1];
     counter=counter+length(find(J~=J1));  
     err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
     totCount = totCount + length(J);
   end
   preds = [preds predVec];

   test_err(epoch)=counter/totCount;
   test_crerr(epoch)=err_cr;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  %%%% TRAINING STATS
  %%%% Error rates
  [numcases numdims numbatches]=size(batchdata);
  N=numcases;
  bias_hid = repmat(h1_biases,N,1);
  bias_pen = repmat(h2_biases,N,1);
  bias_top = repmat(topbiases,N,1);
  err=0;
  err_cr=0;
  counter=0;
  totCount = 0;
  for batch = 1:numbatches
    data = [batchdata(:,:,batch)];
    temp_h2 = temp_h2_train(:,:,batch); 
    target = [batchdata_trait(:,:,batch)];

    w1probs = 1./(1 + exp(-data*w1_vishid -temp_h2*w1_penhid - bias_hid  ));
    w2probs = 1./(1 + exp(-w1probs*w2 - bias_pen));
    targetout = exp(w2probs*w_class + bias_top );
    targetout = targetout./repmat(sum(targetout,2),1,numlab);
    [I J]=max(targetout,[],2);
    [I1 J1]=max(target,[],2);
    counter=counter+length(find(J~=J1));
    err_cr = err_cr- sum(sum( target(:,1:end).*log(targetout))) ;
    totCount = totCount + length(J);
  end

  train_err(epoch)=counter/totCount;
  train_crerr(epoch)=err_cr;

  fprintf(1,'epoch %d: train_err %f, test_err %f; train_crerr %f, test_crerr %f \n',...
      epoch, train_err(epoch), test_err(epoch), train_crerr(epoch), test_crerr(epoch));
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  w1_vishid_all = cat(3,w1_vishid_all,w1_vishid);
  w2_all = cat(3,w2_all,w2);
  w_class_all = cat(3,w_class_all,w_class);
  
  outFile = [outFileTag '_backprop.mat'];
  save(outFile,'w1_vishid','w1_penhid','w2','w_class','h1_biases','h2_biases',...
      'topbiases','test_err','test_crerr','train_err','train_crerr','preds','y_test');
 
  if epoch==maxepoch
      save(outFile,'w1_vishid_all','w2_all','w_class_all','-append');
      break;
  end
  
%%% Do Conjugate Gradient Optimization 

  for batch = 1:numbatches
%     fprintf(1,'epoch %d batch %d\r',epoch,batch);
    
    data = [batchdata(:,:,batch)];
    temp_h2 = temp_h2_train(:,:,batch); 
    targets = [batchdata_trait(:,:,batch)];    

%%%%%%%% DO CG with 3 linesearches 

    VV = [w1_vishid(:)' w1_penhid(:)' w2(:)' w_class(:)' h1_biases(:)' h2_biases(:)' topbiases(:)']';
    Dim = [numdims; numhids; numpens; numlab];

% checkgrad('CG_MNIST_INIT',VV,10^-5,Dim,data,targets);
    max_iter=3; 
    if epoch<6
       [X, fX, num_iter,ecg_XX] = dspn_minimize(VV,'dspn_CG_INIT',max_iter,Dim,data,targets,temp_h2);
    else
       [X, fX, num_iter,ecg_XX] = dspn_minimize(VV,'dspn_CG',max_iter,Dim,data,targets,temp_h2);
    end 
    w1_vishid = reshape(X(1:numdims*numhids),numdims,numhids);
    xxx = numdims*numhids;
    w1_penhid = reshape(X(xxx+1:xxx+numpens*numhids),numpens,numhids);
    xxx = xxx+numpens*numhids;
    w2 = reshape(X(xxx+1:xxx+numhids*numpens),numhids,numpens);
    xxx = xxx+numhids*numpens;
    w_class = reshape(X(xxx+1:xxx+numpens*numlab),numpens,numlab);
    xxx = xxx+numpens*numlab;
    h1_biases = reshape(X(xxx+1:xxx+numhids),1,numhids);
    xxx = xxx+numhids;
    h2_biases = reshape(X(xxx+1:xxx+numpens),1,numpens);
    xxx = xxx+numpens;
    topbiases = reshape(X(xxx+1:xxx+numlab),1,numlab);
    xxx = xxx+numlab;

  end

end


