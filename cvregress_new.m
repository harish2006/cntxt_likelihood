% cross validated regression
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
% cavg: Average per test set correlations between predicted y and observed y.
% pavg: Average significance of correlations per-split
% predy: Concatenated vector containing predicted values of y from each
% test set.
% stats: Detailed statistics from cross validation
%        stats.w : nsplits x (nfeatures+1) regression weight matrix 
%        stats.c : predicted vs observed correlation per-split 
%        stats.p : significance of predicted vs observed correlation per-split 
%        stats.testids : test set indices for each training split
%        stats.rsquared : R^2 statistic returned by regress. Fraction of
%        explainable variance that is captured by regression model fit.

% harish
% 30 Dec 2015

function [cavg,pavg,predy,stats]=cvregress_new(y,X,testfrac,randomize,iterations,refsplits)
    if ~exist('testfrac')
        testfrac=0.2;
    end
    if ~exist('randomize')
        randomize=1;
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
    
    y=vec(y);
    cavg=[];pavg=[];predy=[];
    stats=[];stats.w=[];stats.c=[];stats.p=[];stats.testids=[];stats.rsquared=[];
    nobs=size(X,1);
    
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
            qb=find(isnan(trny));
            trny(qb)=[];
            trnx(qb,:)=[];
            [b,bint,rint,regstats]=regress(trny,trnx);b=b(:);
            predtsty=tstx*b;
            stats.w=[stats.w;b'];
            % correlate y and predy only if two or more test elements
            if length(tstind)>1
                [c,p]=nancorrcoef(predtsty,tsty);
                stats.c=[stats.c;c];
                stats.p=[stats.p;p];
                stats.rsquared=[stats.rsquared;regstats(1)];
            else
                stats.c=[stats.c;nan];
                stats.p=[stats.p;nan];
                stats.rsquared=[stats.rsquared; nan];
            end
            % store image ids for each split
            stats.testids{length(stats.testids)+1}=tstind;
            % pool predictions across splits
            py(tstind)=predtsty;
        end
        predy=[predy py(:)];        
    end
    predy=nanmean(predy,2); % aveage predictions from all iterations.
    % correlated pooled predicted y and observed y
    [cavg,pavg]=nancorrcoef(predy,y);
end