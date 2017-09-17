function t=myzscore(t)
    meant=nanmean(t,1);stdt=nanstd(t,1);
    t=(t-repmat(meant,[size(t,1) 1]))./repmat(stdt,[size(t,1) 1]);
    t(isnan(t))=0;
end