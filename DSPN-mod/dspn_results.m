function dspn_results(inFold)

if nargin==0
    inFold = 'out/';
end
nMod = 10;

errsAll = [];
vecs = [];

unqs = [];
for i = 1:nMod
    
    fname = [inFold 'model' num2str(i) '_backprop.mat'];
    load(fname,'test_err');
    errsAll = [errsAll min(test_err)];
    vecs = [vecs ; test_err];
    unq = length(unique(test_err));
    unqs = [unqs unq];
    
end

acc1 = 1 - mean(errsAll);
vec = mean(vecs);
plot(1:size(vecs,2),vec);
acc2 = 1 - min(vec);

%%%%%%%% liability

vec = mean(vecs);
stop_idx = find(vec==min(vec),1);

K = 0.011;
idx = norminv(K);
z = normpdf(idx);
i_scz = z / K;

est = [];
y = [];
vls = [];
props = [];
for j = 1:nMod 
    fname = [inFold 'model' num2str(j) '_backprop.mat'];
    load(fname,'preds','test_err','y_test');
    idx = find(test_err==min(test_err),1);
    pred = preds(:,idx);
    est = pred;
    y = y_test;
    props = [props mean(est==y)];
    
    est = est - 1; y = y - 1;
    a = sum((est==0)&(y==0)); if a==0, a=1; end
    b = sum((est==0)&(y==1)); if b==0, b=1; end
    c = sum((est==1)&(y==0)); if c==0, c=1; end
    d = sum((est==1)&(y==1)); if d==0, d=1; end
    grr = (d/(c+d)) / (b/(a+b));
    pl = (c + (K/(1-K))*d) / (a + c + (K/(1-K))*(b + d));
    vl = 2 * pl * (1-pl) * ((grr-1)^2) / (i_scz^2);
    vls = [vls vl];    
end
liab1 = mean(vls(isfinite(vls)));

vls = [];
props = [];
for j = 1:nMod 
    fname = [inFold 'model' num2str(j) '_backprop.mat'];
    load(fname,'preds','test_err','y_test');
    idx = stop_idx;
    pred = preds(:,idx);
    est = pred;
    y = y_test;
    props = [props mean(est==y)];
    
    est = est - 1; y = y - 1;
    a = sum((est==0)&(y==0)); if a==0, a=1; end
    b = sum((est==0)&(y==1)); if b==0, b=1; end
    c = sum((est==1)&(y==0)); if c==0, c=1; end
    d = sum((est==1)&(y==1)); if d==0, d=1; end
    grr = (d/(c+d)) / (b/(a+b));
    pl = (c + (K/(1-K))*d) / (a + c + (K/(1-K))*(b + d));
    vl = 2 * pl * (1-pl) * ((grr-1)^2) / (i_scz^2);
    vls = [vls vl];    
end
liab2 = mean(vls(isfinite(vls)));

display('Per-model early stopping:');
display(['accuracy: ' num2str(acc1)]);
display(['liability: ' num2str(liab1)]);
display(' ');

display('Cross-model early stopping:');
display(['accuracy: ' num2str(acc2)]);
display(['liability: ' num2str(liab2)]);

