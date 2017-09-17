% does predicted target likelihood improve discrimination of cnn outputs.
function [car,per]=ratings_improve_cnnperf_ade20K_cntxtpred()
    restriction=1;
    bach=0;
    
    %resutls restricted 
    % rcnn
    %     0.8226    0.8293    0.8239    0.8293    0.8320    0.8333    0.8535
    %     0.8023    0.8023    0.8121    0.8056    0.8088    0.8105    0.8007
    % bach
    %    0.8347    0.8320    0.8347    0.8427    0.8468    0.8401    0.8535
    %    0.7288    0.7288    0.7696    0.7533    0.7565    0.7745    0.7631
    % bach network
    if bach==1
        load ./ade20K/cardet_voc_scores_ade20K;
    else
        load ./ade20K/cardet_rcnn_voc_scores_ade20K;
    end
    
    load ./ade20K/L2_car_ade20k.mat;
    load ./ade20K/restricted_carscenes_ade20K.mat;

    if restriction==1
        % apply restricted set here. select a subset of absent and present
        % scenes.
        L2_car.hastarget=L2_car.hastarget([qgood;qgood]==1);
        L2_car.blurim_placenet_fc7=L2_car.blurim_placenet_fc7([qgood;qgood]==1,:);
        L2_car.blurgist=L2_car.blurgist([qgood;qgood]==1,:);
    end
    
    hastarget_car=L2_car.hastarget;
    L2_ade20K=L2_car;
    
       
	dims=60;

    if bach==1
        nimgs=length(cardet_voc_scores);
        cscores=zeros(nimgs,20);for i=1:nimgs;cscores(i,:)=cardet_voc_scores(i).res{1}';end;
        cnn_car=cscores(:,7); % get car ratings only
    else
        cnn_car=vec(cardet_voc_scores);
    end
    
    if restriction==1
        % apply restricted set
        cnn_car=cnn_car([qgood;qgood]==1);
    end
    
    idpres=1:650;
    idabs=651:1300;
    % load human ratings
    load L2_car;

  
    [c,s,l]=princomp([L2_car.blurim_placenet_fc7; L2_ade20K.blurim_placenet_fc7]);
    cntxt=s(:,1:dims); 
    feat=cntxt(1:1300,:); % separate out the two datasets.
    featade20K=cntxt(1301:end,:);
 	[c,s,l]=princomp([L2_car.blurgist; L2_ade20K.blurgist]);
    cntxt=s(:,1:dims); 
    feat=[feat cntxt(1:1300,:)]; % separate out the two datasets.
    featade20K=[featade20K cntxt(1301:end,:)];

    % perform car classification using coarse features on ade20K
%     [predy pcm coefs post]=cvclassify([featade20K ones(size(featade20K,1))],vec(L2_ade20K.hastarget),0.2); % cnxt class preds
%     L2_car.pcar=vec(post(:,2));
    
    car=[];
    car.predlklhd=cnn_analysis_predratings_ade20K(L2_car.pcar,feat,idabs,idpres,featade20K);
    car.predlocn=cnn_analysis_predratings_ade20K(L2_car.rectcar(:,2),feat,idabs,idpres,featade20K);
    car.predscale=cnn_analysis_predratings_ade20K(L2_car.rectcar(:,3).*L2_car.rectcar(:,4),feat,idabs,idpres,featade20K);
    ar=L2_car.rectcar(:,3)./L2_car.rectcar(:,4);
    car.predar=cnn_analysis_predratings_ade20K(ar,feat,idabs,idpres,featade20K);
    
    % also predict ratings for other category, maybe they are informative
    car.predlklhdper=cnn_analysis_predratings_ade20K(L2_car.pperson,feat,idabs,idpres,featade20K);
    car.predlocnper=cnn_analysis_predratings_ade20K(L2_car.rectperson(:,2),feat,idabs,idpres,featade20K);
    car.predscaleper=cnn_analysis_predratings_ade20K(L2_car.rectperson(:,3).*L2_car.rectperson(:,4),feat,idabs,idpres,featade20K);
    car.predarper=cnn_analysis_predratings_ade20K(L2_car.rectperson(:,4)./L2_car.rectperson(:,3),feat,idabs,idpres,featade20K);
    % now for classifier analysis using cnn confidence alone,
    % models containing one of the ratings as well
    % combined model containing all three
%     cnn_car=zscore(cnn_car);
%     car.predlklhd=zscore(car.predlklhd);
%     car.predlocn=zscore(car.predlocn);
%     car.predscale=zscore(car.predscale);
    
    car.acc=[]; % cnn only, cnn+lklhd, cnn+ y-locn, cnn+area (scale), cnn+ all three ratings
%     [class,pcm]=looclassify(cnn_car,vec(hastarget_car));car.acc(1)=sum(class==vec(hastarget_car))/length(vec(hastarget_car));
%     [class,pcm]=looclassify([cnn_car car.predlklhd],vec(hastarget_car));car.acc(2)=sum(class==vec(hastarget_car))/length(vec(hastarget_car));
%     [class,pcm]=looclassify([cnn_car car.predlocn],vec(hastarget_car));car.acc(3)=sum(class==vec(hastarget_car))/length(vec(hastarget_car));
%     [class,pcm]=looclassify([cnn_car car.predscale],vec(hastarget_car));car.acc(4)=sum(class==vec(hastarget_car))/length(vec(hastarget_car));
%     [class,pcm]=looclassify([cnn_car car.predlklhd car.predlocn car.predscale],vec(hastarget_car));car.acc(5)=sum(class==vec(hastarget_car))/length(vec(hastarget_car));

%     gt=[ones(650,1);zeros(650,1)];
%     car.acc(1)=cvregress_new(gt,[cnn_car ones(1300,1)],0.2);
%     car.acc(2)=cvregress_new(gt,[cnn_car car.predlklhd ones(1300,1)],0.2);
%     car.acc(3)=cvregress_new(gt,[cnn_car car.predlocn ones(1300,1)],0.2);
%     car.acc(4)=cvregress_new(gt,[cnn_car car.predscale ones(1300,1)],0.2);
%     car.acc(5)=cvregress_new(gt,[cnn_car car.predlklhd car.predlocn car.predscale ones(1300,1)],0.2);

    [class car.pcm{1}]=cvclassify(cnn_car,vec(hastarget_car));
    car.acc(1)=car.pcm{1}(1,1)+car.pcm{1}(2,2);
    [class car.pcm{2}]=cvclassify([cnn_car car.predlklhd],vec(hastarget_car));
    car.acc(2)=car.pcm{2}(1,1)+car.pcm{2}(2,2);
    [class car.pcm{3}]=cvclassify([cnn_car car.predlocn],vec(hastarget_car));
    car.acc(3)=car.pcm{3}(1,1)+car.pcm{3}(2,2);
    [class car.pcm{4}]=cvclassify([cnn_car car.predscale],vec(hastarget_car))
    car.acc(4)=car.pcm{4}(1,1)+car.pcm{4}(2,2);
    [class car.pcm{5}]=cvclassify([cnn_car car.predlklhd car.predlocn car.predscale],vec(hastarget_car));
    car.acc(5)=car.pcm{5}(1,1)+car.pcm{5}(2,2);
    [class car.pcm{6}]=cvclassify([cnn_car car.predlklhd car.predlocn car.predscale car.predar],vec(hastarget_car));
    car.acc(6)=car.pcm{6}(1,1)+car.pcm{6}(2,2);
    
    [class car.pcm{7}]=cvclassify([cnn_car car.predlklhd car.predlocn car.predscale car.predar car.predlklhdper car.predlocnper car.predscaleper car.predarper],vec(hastarget_car));
    car.acc(7)=car.pcm{7}(1,1)+car.pcm{7}(2,2);
    %%
    figure(101);
    subplot(241)
    %fit_n_draw_line(cnn_car(hastarget_car==1),car.predlklhd(hastarget_car==1));
    hold on;
    plot(cnn_car(hastarget_car==1),car.predlklhd(hastarget_car==1),'r.');
    plot(cnn_car(hastarget_car==0),car.predlklhd(hastarget_car==0),'b.');
    xlabel('cnn car confidence');ylabel('predicted car likelihood');
    legend({'car present','car absent'});
	
    subplot(242)
    %fit_n_draw_line(cnn_car(hastarget_car==1),car.predlocn(hastarget_car==1));
    hold on;
    plot(cnn_car(hastarget_car==1),car.predlocn(hastarget_car==1),'r.');
    plot(cnn_car(hastarget_car==0),car.predlocn(hastarget_car==0),'b.');
    xlabel('cnn car confidence');ylabel('predicted car y-locn');
    legend({'car present','car absent'});

    subplot(243)
    %fit_n_draw_line(cnn_car(hastarget_car==1),car.predscale(hastarget_car==1));
    hold on;
    plot(cnn_car(hastarget_car==1),car.predscale(hastarget_car==1),'r.');
    plot(cnn_car(hastarget_car==0),car.predscale(hastarget_car==0),'b.');
    xlabel('cnn car scale');ylabel('predicted car area');
    legend({'car present','car absent'});

    subplot(244)
    %fit_n_draw_line(cnn_car(hastarget_car==1),car.predscale(hastarget_car==1));
    hold on;
    plot(cnn_car(hastarget_car==1),car.predar(hastarget_car==1),'r.');
    plot(cnn_car(hastarget_car==0),car.predar(hastarget_car==0),'b.');
    xlabel('cnn car aspratio');ylabel('predicted car area');
    legend({'car present','car absent'});
    
    %%
    % bach network
    if bach==1
        load ./ade20K/perdet_voc_scores_ade20K.mat;
    else
    % rcnn network
        load ./ade20K/perdet_rcnn_voc_scores_ade20K;
    end
    
    load ./ade20K/L2_per_ade20k.mat;
	

	% apply restricted set here. select a subset of absent and present
    % scenes.
    load ./ade20K/restricted_personscenes_ade20K.mat;
    
    if restriction==1
        L2_per.hastarget=L2_per.hastarget([qgood;qgood]==1);
        L2_per.blurim_placenet_fc7=L2_per.blurim_placenet_fc7([qgood;qgood]==1,:);
        L2_per.blurgist=L2_per.blurgist([qgood;qgood]==1,:);
    end
    
    hastarget_per=L2_per.hastarget;
    L2_ade20K=L2_per;
    
    
    if bach==1
        nimgs=length(perdet_voc_scores);
        pscores=zeros(nimgs,20);for i=1:nimgs;pscores(i,:)=perdet_voc_scores(i).res{1}';end;
        cnn_per=pscores(:,15); % get person ratings only
    else
        cnn_per=vec(perdet_voc_scores);
    end
    
    if restriction==1
        % apply restricted set
        cnn_per=cnn_per([qgood;qgood]==1);
    end
    
    idpres=1:650;
    idabs=651:1300;
    % load human ratings
    load L2_per;

    [c,s,l]=princomp([L2_per.blurim_placenet_fc7; L2_ade20K.blurim_placenet_fc7]);
    cntxt=s(:,1:dims); 
    feat=cntxt(1:1300,:); % separate out the two datasets.
    featade20K=cntxt(1301:end,:);
    [c,s,l]=princomp([L2_per.blurgist; L2_ade20K.blurgist]);
    cntxt=s(:,1:dims); 
    feat=[feat cntxt(1:1300,:)]; % separate out the two datasets.
    featade20K=[featade20K cntxt(1301:end,:)];
    
%     % perform person classification using coarse features on ade20K
%     [predy pcm coefs post]=cvclassify([featade20K ones(size(featade20K,1))],vec(L2_ade20K.hastarget),0.2); % cnxt class preds
%     L2_per.pperson=vec(post(:,2));
    
    per=[];
    per.predlklhd=cnn_analysis_predratings_ade20K(L2_per.pperson,feat,idabs,idpres,featade20K);
    per.predlocn=cnn_analysis_predratings_ade20K(L2_per.rectperson(:,2),feat,idabs,idpres,featade20K);
    per.predscale=cnn_analysis_predratings_ade20K(L2_per.rectperson(:,3).*L2_per.rectperson(:,4),feat,idabs,idpres,featade20K);
    per.predar=cnn_analysis_predratings_ade20K(L2_per.rectperson(:,3)./L2_per.rectperson(:,4),feat,idabs,idpres,featade20K);
    
    % also predict ratings for other category
	per.predlklhdcar=cnn_analysis_predratings_ade20K(L2_per.pcar,feat,idabs,idpres,featade20K);
    per.predlocncar=cnn_analysis_predratings_ade20K(L2_per.rectcar(:,2),feat,idabs,idpres,featade20K);
    per.predscalecar=cnn_analysis_predratings_ade20K(L2_per.rectcar(:,3).*L2_per.rectcar(:,4),feat,idabs,idpres,featade20K);
    per.predarcar=cnn_analysis_predratings_ade20K(L2_per.rectcar(:,3)./L2_per.rectcar(:,4),feat,idabs,idpres,featade20K);


%     cnn_per=zscore(cnn_per);
%     per.predlklhd=zscore(per.predlklhd);
%     per.predlocn=zscore(per.predlocn);
%     per.predscale=zscore(per.predscale);
    
    % now for classifier analysis using cnn confidence alone,
    % models containing one of the ratings as well
    % combined model containing all three
    per.acc=[]; % cnn only, cnn+lklhd, cnn+ y-locn, cnn+area (scale), cnn+ all three ratings
%     [class,pcm]=looclassify(cnn_per,vec(hastarget_per));per.acc(1)=sum(class==vec(hastarget_per))/length(vec(hastarget_per));
%     [class,pcm]=looclassify([cnn_per per.predlklhd],vec(hastarget_per));per.acc(2)=sum(class==vec(hastarget_per))/length(vec(hastarget_per));
%     [class,pcm]=looclassify([cnn_per per.predlocn],vec(hastarget_per));per.acc(3)=sum(class==vec(hastarget_per))/length(vec(hastarget_per));
%     [class,pcm]=looclassify([cnn_per per.predscale],vec(hastarget_per));per.acc(4)=sum(class==vec(hastarget_per))/length(vec(hastarget_per));
%     [class,pcm]=looclassify([cnn_per per.predlklhd per.predlocn per.predscale],vec(hastarget_per));per.acc(5)=sum(class==vec(hastarget_per))/length(vec(hastarget_per));

%     gt=[ones(650,1);zeros(650,1)];
%     per.acc(1)=cvregress_new(gt,[cnn_per ones(1300,1)],0.2);
%     per.acc(2)=cvregress_new(gt,[cnn_per per.predlklhd ones(1300,1)],0.2);
%     per.acc(3)=cvregress_new(gt,[cnn_per per.predlocn ones(1300,1)],0.2);
%     per.acc(4)=cvregress_new(gt,[cnn_per per.predscale ones(1300,1)],0.2);
%     per.acc(5)=cvregress_new(gt,[cnn_per per.predlklhd per.predlocn per.predscale ones(1300,1)],0.2);

    [class per.pcm{1}]=cvclassify(cnn_per,vec(hastarget_per));
    per.acc(1)=per.pcm{1}(1,1)+per.pcm{1}(2,2);
    [class per.pcm{2}]=cvclassify([cnn_per per.predlklhd],vec(hastarget_per));
    per.acc(2)=per.pcm{2}(1,1)+per.pcm{2}(2,2);
    [class per.pcm{3}]=cvclassify([cnn_per per.predlocn],vec(hastarget_per));
    per.acc(3)=per.pcm{3}(1,1)+per.pcm{3}(2,2);
    [class per.pcm{4}]=cvclassify([cnn_per per.predscale],vec(hastarget_per));
    per.acc(4)=per.pcm{4}(1,1)+per.pcm{4}(2,2);
    [class per.pcm{5}]=cvclassify([cnn_per per.predlklhd per.predlocn per.predscale],vec(hastarget_per));
    per.acc(5)=per.pcm{5}(1,1)+per.pcm{5}(2,2);
    [class per.pcm{6}]=cvclassify([cnn_per per.predlklhd per.predlocn per.predscale per.predar],vec(hastarget_per));
    per.acc(6)=per.pcm{6}(1,1)+per.pcm{6}(2,2);
    
    [class per.pcm{7}]=cvclassify([cnn_per per.predlklhd per.predlocn per.predscale per.predar per.predlklhdcar per.predlocncar per.predscalecar per.predarcar],vec(hastarget_per));
    per.acc(7)=per.pcm{7}(1,1)+per.pcm{7}(2,2);

    %%
    figure(101);
    subplot(245)
    %fit_n_draw_line(cnn_per(hastarget_per==1),per.predlklhd(hastarget_per==1));
    hold on;
    plot(cnn_per(hastarget_per==1),per.predlklhd(hastarget_per==1),'r.');
    plot(cnn_per(hastarget_per==0),per.predlklhd(hastarget_per==0),'b.');
    xlabel('cnn person confidence');ylabel('predicted person likelihood');
    legend({'person present','person absent'});
	
    subplot(246)
    %fit_n_draw_line(cnn_per(hastarget_per==1),per.predlocn(hastarget_per==1));
    hold on;
    plot(cnn_per(hastarget_per==1),per.predlocn(hastarget_per==1),'r.');
    plot(cnn_per(hastarget_per==0),per.predlocn(hastarget_per==0),'b.');
    xlabel('cnn person confidence');ylabel('predicted person y-locn');
    legend({'person present','person absent'});

    subplot(247)
    %fit_n_draw_line(cnn_per(hastarget_per==1),per.predscale(hastarget_per==1));
    hold on;
    plot(cnn_per(hastarget_per==1),per.predscale(hastarget_per==1),'r.');
    plot(cnn_per(hastarget_per==0),per.predscale(hastarget_per==0),'b.');
    xlabel('cnn person scale');ylabel('predicted person area');
    legend({'person present','person absent'});
    
    subplot(248)
    %fit_n_draw_line(cnn_per(hastarget_per==1),per.predscale(hastarget_per==1));
    hold on;
    plot(cnn_per(hastarget_per==1),per.predar(hastarget_per==1),'r.');
    plot(cnn_per(hastarget_per==0),per.predar(hastarget_per==0),'b.');
    xlabel('cnn person aspratio');ylabel('predicted person area');
    legend({'person present','person absent'});
    %%
    % plot the results of classifier analysis
    figure(102);
    title('classifier analysis');
    bar([car.acc;per.acc]);
    set(gca,'XTickLabel',{'Car classification','Person classification'});
    legend({'CNN','CNN + pred-lklhd','CNN + pred-yloc','CNN + pred-scale','CNN + pred-ar','CNN + pred-all'});
    ylabel('target present/absent accuracy');
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
    
%     [c,s,l]=princomp([L2.blurgist]);
%     cntxt=[cntxt s(:,1:2:2*dims)];    
end