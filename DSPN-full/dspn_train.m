function dspn_train

datFold = 'datasets/';
tag = 'scz';
qt = 1;
numhid = 400;   %-- number of hidden units
numpen = 100;

nPerBatch = 640; % prop
nBatch = 1;
maxepoch = 50;  %-- maximum number of epochs

errsAll = [];

fid = fopen([datFold 'XYgenes.txt'],'r');
XY = textscan(fid,'%s');
XY = XY{1};
fclose(fid);

for modelIt = 1:10

    rng(1000);    
    
    numlab = 2;
    batchdata = [];
    testdata = [];
    inFile = [datFold tag '_data' num2str(modelIt) '.mat'];
    outFileTag = ['out/model' num2str(modelIt)];
    nMF = 20; % for imputt
    
    warning off;
    load(inFile,'nData','nData_te','nSNP','nTrait','X_sampIds_tr','X_sampIds_te','X_geneIds',...
            'X_Gene_tr','X_Gene_te','X_Gene2_tr','X_Gene2_te',...
            'X_Trait_tr','X_Trait_te','X_geneIds2','sc');
        
    if ~exist('nData_te')
        nData_te = nData;
    end        
        
    outFile = [strrep(outFileTag,'out','init') '_init.mat'];
    load(outFile,'labpen', 'labbiases', 'hidpen', 'penbiases', 'vishid', 'hidbiases',...
        'visbiases', 'snpvis');
    
    X_SNP_tr = zeros(nData,2);
    X_SNP_te = zeros(nData_te,2);
    nSNP = 2;
    
    filt = ~ismember(X_geneIds2,XY);
    X_Gene_tr = X_Gene_tr(:,filt);
    X_Gene_te = X_Gene_te(:,filt);
    sc = sc(filt);
    
    batchdata = [];
    batchdata_snp = [];
    batchdata_trait = [];
    traindata = X_Gene_tr(1:nPerBatch*nBatch,:);
    traindata_snp = X_SNP_tr(1:nPerBatch*nBatch,:);
    traindata_trait = X_Trait_tr(1:nPerBatch*nBatch,:);
    for i = 1:nBatch
        batchdata = cat(3,batchdata,X_Gene_tr(1:nPerBatch,:));
        batchdata_snp = cat(3,batchdata_snp,X_SNP_tr(1:nPerBatch,:));
        batchdata_trait = cat(3,batchdata_trait,X_Trait_tr(1:nPerBatch,:));
        X_Gene_tr = X_Gene_tr(nPerBatch+1:end,:);
        X_Trait_tr = X_Trait_tr(nPerBatch+1:end,:);
        X_SNP_tr = X_SNP_tr(nPerBatch+1:end,:);
    end
    
    testdata = X_Gene_te;
    testdata_snp = X_SNP_te;
    testdata_trait = X_Trait_te;

    X_train0 = traindata;
    y_train = traindata_trait(:,2);
    nGene = size(X_train0,2);
    cut = ceil(nGene*(qt/100));
    [dum pmt] = sort(abs(sc),'descend');
    traindata = traindata(:,pmt);
    batchdata = batchdata(:,pmt,:);
    testdata = testdata(:,pmt);
    traindata = traindata(:,1:cut);
    batchdata = batchdata(:,1:cut,:);
    testdata = testdata(:,1:cut);    
    
    dspn_backprop;
    
    errsAll = [errsAll min(test_err)];
    
end
