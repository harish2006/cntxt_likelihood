% script to get model goodness ratings using AICC
% Model name	dof	Person detection	Person rejection	Car detection	Car rejection
%     		r	AICc                r	AICc            r	AICc        r	AICc

% harish, 10th April 2016
function stats=table2_aicc_analysis_cv_nreps_likelhood_new_wgist(nreps,dims)
    useRT=2;
    % this is a wrapper function that generates one table for each of car
    % and persont task
    stats=[];
%     nreps=100;
%     useRT=2;
    fprintf('correlation and aicc analysis\n');
    rvals=pertask_rvals(dims,nreps);
    %%
    modelcomb={'cho','h','o','c','ho','hc','oc','h*3','o*3','c*3'};
    fprintf('Models predict car ratings\n');
    for i=1:size(rvals.pcar,1)
        % present model, present->absent, present model rank
        % absent model, absent->present, absent model rank
        fprintf('%s,%1.2f(%1.2f),%1.2f(%1.2f),%1.2f(%1.2f),%1.2f(%1.2f),%1.2f(%1.2f),%1.2f(%1.2f)\n',...
            modelcomb{i},...
            nanmean(rvals.pcar(i,:)),nanstd(rvals.pcar(i,:)),...
            nanmean(rvals.rarank_car(i,:)),nanstd(rvals.rarank_car(i,:)),...
            nanmean(rvals.xcar(i,:)),nanstd(rvals.xcar(i,:)),...
            nanmean(rvals.ycar(i,:)),nanstd(rvals.ycar(i,:)),...
            nanmean(rvals.areacar(i,:)),nanstd(rvals.areacar(i,:)),...
            nanmean(rvals.aspcar(i,:)),nanstd(rvals.aspcar(i,:)));
    end
    fprintf('Models predict person ratings\n');
    for i=1:size(rvals.pper,1)
        % present model, present->absent, present model rank
        % absent model, absent->present, absent model rank
        fprintf('%s,%1.2f(%1.2f),%1.2f(%1.2f),%1.2f(%1.2f),%1.2f(%1.2f),%1.2f(%1.2f),%1.2f(%1.2f)\n',...
            modelcomb{i},...
            nanmean(rvals.pper(i,:)),nanstd(rvals.pper(i,:)),...
            nanmean(rvals.rarank_per(i,:)),nanstd(rvals.rarank_per(i,:)),...
            nanmean(rvals.xper(i,:)),nanstd(rvals.xper(i,:)),...
            nanmean(rvals.yper(i,:)),nanstd(rvals.yper(i,:)),...
            nanmean(rvals.areaper(i,:)),nanstd(rvals.areaper(i,:)),...
            nanmean(rvals.aspper(i,:)),nanstd(rvals.aspper(i,:)));
    end
    %%
    fname=['rvals_likelihood_' num2str(nreps) '_wgist.mat'];
    save(fname,'rvals');
end
% (10xnreps): r_aicc.rpres  , r_aicc.rpresstd, r_aicc.rpresrank, r_aicc.rabs, r_aicc.rabsstd, r_aicc.rabsrank
function rvals=pertask_rvals(dims,nreps)
    idtarget=1:650;
    idabs=651:1300;
    
%     load L2_car;
%     % remove high standard deviation images
%     stdper=nan(1300,2);for i=651:1300; stdper(i,1)=nanstd(L2_car.rectpersonraw(i,:,1),[],2);stdper(i,2)=nanstd(L2_car.rectpersonraw(i,:,2),[],2);end;
%     stdcar=nan(1300,2);for i=651:1300; stdcar(i,1)=nanstd(L2_car.rectcarraw(i,:,1),[],2);stdcar(i,2)=nanstd(L2_car.rectcarraw(i,:,2),[],2);end;
%     idabs=setdiff(idabs,union(find(stdcar(:,1)>100),find(stdper(:,1)>100)));
    
    rvals=[];
    rvals.pcar=nan(7,nreps);
    rvals.xcar=nan(7,nreps);
    rvals.ycar=nan(7,nreps);
    rvals.areacar=nan(7,nreps);
    rvals.aspcar=nan(7,nreps);
    rvals.pper=nan(7,nreps);
    rvals.xper=nan(7,nreps);
    rvals.yper=nan(7,nreps);
    rvals.areaper=nan(7,nreps);
     rvals.aspper=nan(7,nreps);

    load L2_per.mat;
    L2=L2_per;
    idabs=651:1300;
    
    y.pcar=L2_per.pcar(idabs);
    y.xcar=L2_per.rectcar(idabs,1);
    y.ycar=L2_per.rectcar(idabs,2);
    y.areacar=L2_per.rectcar(idabs,3).*L2_per.rectcar(idabs,4);
    y.aspcar=L2_per.rectcar(idabs,3)./L2_per.rectcar(idabs,4);
    
    y.pper=L2_per.pperson(idabs);
    y.xper=L2_per.rectperson(idabs,1);
    y.yper=L2_per.rectperson(idabs,2);
    y.areaper=L2_per.rectperson(idabs,3).*L2_per.rectperson(idabs,4);
    y.aspper=L2_per.rectperson(idabs,3)./L2_per.rectperson(idabs,4);

    % get features 
    [hogf,tagf,cntxt]=get_feat(L2,dims);
    
    modelcomb={'cho','h','o','c','ho','hc','oc'};
    randomize=0;
    cvfrac=0.2;
    whichsubjs=[];
    for r=1:nreps
        % generate one reference split
        xx=randperm(650);
        for spl=1:5
            reftest{spl}=vec(xx(spl:5:end)); % this is the test set for current split.
        end
        for i=1:length(modelcomb)
            npars=dims*length(modelcomb{i});
            % setup appropriate feature matrix
            feat=[];
            if ~isempty(strfind(modelcomb{i},'h'))
                feat=[feat hogf];
            end
            if ~isempty(strfind(modelcomb{i},'o'))
                feat=[feat tagf];
            end
            if ~isempty(strfind(modelcomb{i},'c'))
                feat=[feat cntxt];
            end
            feat=zscore(feat);        
            % build models for absent scenes
            X=feat(idabs,:);X=[X ones(size(X,1),1)];
            
            % predict various car ratings
            rvals.pcar(i,r)=cvregress_new(y.pcar,X,cvfrac,0,1,reftest);
            rvals.xcar(i,r)=cvregress_new(y.xcar,X,cvfrac,0,1,reftest);
            rvals.ycar(i,r)=cvregress_new(y.ycar,X,cvfrac,0,1,reftest);
            rvals.areacar(i,r)=cvregress_new(y.areacar,X,cvfrac,0,1,reftest);
            rvals.aspcar(i,r)=cvregress_new(y.aspcar,X,cvfrac,0,1,reftest);
            
            % predict various person ratings
            rvals.pper(i,r)=cvregress_new(y.pper,X,cvfrac,0,1,reftest);
            rvals.xper(i,r)=cvregress_new(y.xper,X,cvfrac,0,1,reftest);
            rvals.yper(i,r)=cvregress_new(y.yper,X,cvfrac,0,1,reftest);
            rvals.areaper(i,r)=cvregress_new(y.areaper,X,cvfrac,0,1,reftest);
            rvals.aspper(i,r)=cvregress_new(y.aspper,X,cvfrac,0,1,reftest);
        end
        % now assign rank based on how well model predicts pcar
        [~,ind]=sort(rvals.pcar(:,r),'descend');
        for rnk=1:7
            rvals.rarank_car(rnk,r)=find(ind==rnk);
        end
        [~,ind]=sort(rvals.pper(:,r),'descend');
        for rnk=1:7
            rvals.rarank_per(rnk,r)=find(ind==rnk);
        end
        fprintf('finished rep %d \n',r);
    end
end

function [hogf,tagf,cntxt]=get_feat(L2,dims)
   
    % features in the order target, nontarget, coarse context
    % this version can pick and choose from different model types
    feat=[];

    hogf=L2.dpmhog;
    [c,s,l]=princomp(hogf);     
    hogf=s(:,1:dims);   
    
    t=10;
    b=find(nansum(L2.tagfreq,1)>t);
    b=setdiff(b,[11 14 40 44 48 60]);
    % exclude following concepts
    %     11 colour 
    %     14 thing 
    %     40 shadow 
    %     44 lights 
    %     48 bright 
    %     60 shelter 

    [c,s,l]=princomp(L2.tagfreq(:,b));
    tagf=s(:,1:dims); 
    
    % adding gist to the coarse structure signature.
    [c,s,l]=princomp([L2.blurim_placenet_fc7]);
    cntxt=s(:,1:dims); 
    
    [c,s,l]=princomp([L2.blurgist]);
    cntxt=[cntxt s(:,1:2:2*dims)];    
end