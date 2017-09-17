% cross validated linear classification
% inputs:
% ========
% x : nobs x nfeatures matrix. Last column must be ones for regress() to
% work properly.
% y : nobs x 1 vector of variable to be predicted
% testfrac: testfrac = 0. Leave one out cross validation
%           0 < testfrac < 1: (1-frac)*nobs for train and frac*obs for test.
%           testfrac = 1. Train on entire data, predict entire data
%           defaults to 0.2 or 5-fold cross validation (20% test set).
% randomize: 0- do not randomize/1- randomize order of observations before splitting. 
%           defaults to 1-randomize observations in test/train. Useful when
%           different feature sets on the same set of observations, need to
%           be cross validated using exactly same splits.
% iterations: Number of times a randomized cross validation should run.
% refsplits: if we want specific splits
% outputs:
% ========
% predy: Concatenated vector containing predicted class labels

% harish
% 30 Dec 2015
% 2nd Nov, changed to cross validated classification
function [predy pcm coefs posterior]=cvclassify(X,y,testfrac,randomize,iterations,refsplits)
    if ~exist('testfrac')
        testfrac=0.2;
    end
    if ~exist('randomize')
        randomize=0;
    end
    if ~exist('iterations')
        iterations=50;
    end
    if ~exist('refsplits')
        refsplits=[];
    end
    
    if randomize==0
        iterations=1;
    end
    
    % remove any rows with invalid values
    xx=sum(X,2);
    qgood=~isnan(xx) & ~isinf(xx);
    X=X(qgood,:);
    y=y(qgood,:);
    
    % zscore the columns
    X=zscore(X);
    
    y=vec(y);
    cavg=[];pavg=[];predy=[];posterior=[];
    stats=[];stats.w=[];stats.c=[];stats.p=[];stats.testids=[];stats.rsquared=[];
    nobs=size(X,1);

%     y=y(randperm(length(y)));
    
    for i=1:iterations
        py=[]; % predictions from one iteration
        % cross validation using random sampling w/o replacement ?
        % let it default to randomize == 1 in most cases.
        if randomize==1
            r=vec(randperm(nobs));
        else
            r=vec(1:nobs);
        end

        if isempty(refsplits)
            splits=[];nsplits=[];
            if testfrac==0
                % leave one out cross validation
                nsplits=nobs;
                for i=1:nobs
                    splits{i}=i;
                end;
            elseif testfrac==1
                % use all train set in test set
                nsplits=1;
                splits{1}=1:nobs;
            else
                % use testfrac to define test splits
                nsplits=0;
                nfolds=floor(1/testfrac);
                for i=1:nfolds
                    nsplits= nsplits+1;
                    splits{nsplits}=r(i:nfolds:end);
                end
            end
        else
            splits=refsplits;
            nsplits=length(refsplits);
        end
        % perform regression over all splits.
        for i=1:nsplits
            tstind=splits{i};
            tstx=X(tstind,:);tsty=y(tstind);
            trnind=setdiff(1:nobs,tstind);
            if isempty(trnind)
                % if test set consists of all observations, then use the same
                % for training as well.
                trnind=1:nobs;
            end
            trnx=X(trnind,:);trny=y(trnind);
%             [predtsty,err,p,logp,coef]=classify(tstx,trnx,trny);
            [predtsty,err,p,trnacc,coef]=pcaclassify(tstx,trnx,trny);
            coefs{i}=coef;
%             svmStruct = svmtrain(trnx,trny);
%             predtsty=svmclassify(svmStruct,tstx);
            py(tstind)=predtsty;
            posterior(tstind,:)=p;
        end
        predy=[predy py(:)];  
    end
    predy=nanmean(predy,2); % aveage predictions from all iterations. In classical n-fold the splits will be nonoverlapping
    
    % get hits, false alarms [hits, false alarms;misses true-negatives]
    pcm(1,1)= sum(predy==1 & y==1);
    pcm(1,2)= sum(predy==1 & y==0);
    pcm(2,1)= sum(predy==0 & y==1);
    pcm(2,2)= sum(predy==0 & y==0);
    pcm=pcm/length(y); % get fractions.
end