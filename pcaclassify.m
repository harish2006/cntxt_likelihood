function [class,err,post,trnacc,coef]=pcaclassify(xtst,xtrn,ytrn)
    [c,s,l]=pca(zscore([xtrn;xtst]));
    ndims=find(cumsum(l)/sum(l)>0.9);
    
    dims=ndims(1);

    nxtrn=s(1:size(xtrn,1),1:dims);
    nxtst=s(size(xtrn,1)+1:end,1:dims);

    [class,err,post,coef]=classify(nxtst,nxtrn,ytrn);
    [classtrn,tt,ttt]=classify(nxtrn,nxtrn,ytrn);
    trnacc=sum(classtrn==ytrn(:))/length(ytrn);
end