allclear;close all;
load rvals_likelihood_100_wgist.mat;
% compute error bars, using only L2_cp for the car and person boxes
load L2_car;
load('L2_cp.mat');
id1024=L2_car.id1024(651:1290);
% also put in the noise ceiling on the training model
idabs=651:1300;
[c,~,~,stats.pcar.sem]=mysplithalfcorr(L2_car.pcarraw(:,idabs),100);
stats.pcar.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.xcar.sem]=mysplithalfcorr(squeeze(L2_cp.rectcar(id1024,:,1))',100);
stats.xcar.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.ycar.sem]=mysplithalfcorr(squeeze(L2_cp.rectcar(id1024,:,2))',100);
stats.ycar.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.areacar.sem]=mysplithalfcorr(squeeze(L2_cp.rectcar(id1024,:,3).*L2_cp.rectcar(id1024,:,4))',100);
stats.areacar.sbc=spearmanbrowncorrection(c,2);
xx=squeeze(L2_cp.rectcar(id1024,:,3)./L2_cp.rectcar(id1024,:,4))';
xx(:,find(isnan(nanmean(xx,1))))=[];
[c,~,~,stats.aspcar.sem]=mysplithalfcorr(xx,100);
stats.aspcar.sbc=spearmanbrowncorrection(c,2);

[c,~,~,stats.pper.sem]=mysplithalfcorr(L2_car.ppersonraw(:,idabs),100);
stats.pper.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.xper.sem]=mysplithalfcorr(squeeze(L2_cp.rectperson(id1024,:,1))',100);
stats.xper.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.yper.sem]=mysplithalfcorr(squeeze(L2_cp.rectperson(id1024,:,2))',100);
stats.yper.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.areaper.sem]=mysplithalfcorr(squeeze(L2_cp.rectperson(id1024,:,3).*L2_cp.rectperson(id1024,:,4))',100);
stats.areaper.sbc=spearmanbrowncorrection(c,2);
xx=squeeze(L2_cp.rectperson(id1024,:,3)./L2_cp.rectperson(id1024,:,4))';
[c,~,~,stats.aspper.sem]=mysplithalfcorr(xx,100);
stats.aspper.sbc=spearmanbrowncorrection(c,6);
% 
% rvals_warea=rvals;
% 
% load rvals_likelihood_100_wgist_wasp.mat;
% rvals.aspcar=rvals.areacar;
% rvals.aspper=rvals.areaper;
% 
% rvals.areacar=rvals_warea.areacar;
% rvals.areaper=rvals_warea.areaper;

%%
figure(201);
subplot(121);
barwitherr([nanstd(rvals.pcar(1,:)) nanstd(rvals.xcar(1,:)) nanstd(rvals.ycar(1,:)) nanstd(rvals.areacar(1,:)) nanstd(rvals.aspcar(1,:))],...
    [nanmean(rvals.pcar(1,:)) nanmean(rvals.xcar(1,:)) nanmean(rvals.ycar(1,:)) nanmean(rvals.areacar(1,:)) nanmean(rvals.aspcar(1,:))]);
title('NC predicts car ratings');
hold on;
errorbar([1 2 3 4 5],...
    [stats.pcar.sbc stats.xcar.sbc stats.ycar.sbc stats.areacar.sbc stats.aspcar.sbc],...
    [stats.pcar.sem stats.xcar.sem stats.ycar.sem stats.areacar.sem stats.aspcar.sem],'.'); % standard deviation of noise ceiling
hold on;
ylim([0 1]);
fprintf('noise ceilings pcar %f(%f), xcar %f(%f), ycar %f(%f), areacar %f(%f)\n',...
    stats.pcar.sbc,stats.pcar.sem,...
    stats.xcar.sbc,stats.xcar.sem,...
    stats.ycar.sbc,stats.ycar.sem,...
    stats.areacar.sbc,stats.areacar.sem);
subplot(122);
barwitherr([nanstd(rvals.pper(1,:)) nanstd(rvals.xper(1,:)) nanstd(rvals.yper(1,:)) nanstd(rvals.areaper(1,:)) nanstd(rvals.aspper(1,:))],...
    [nanmean(rvals.pper(1,:)) nanmean(rvals.xper(1,:)) nanmean(rvals.yper(1,:)) nanmean(rvals.areaper(1,:)) nanmean(rvals.aspper(1,:))]);
hold on;
errorbar([1 2 3 4 5],...
    [stats.pper.sbc stats.xper.sbc stats.yper.sbc stats.areaper.sbc stats.aspper.sbc],...
    [stats.pper.sem stats.xper.sem stats.yper.sem stats.areaper.sem stats.aspper.sem],'.'); % standard deviation of noise ceiling
hold on;
ylim([0 1]);
fprintf('noise ceilings pper %f(%f), xper %f(%f), yper %f(%f), areaper %f(%f)\n',...
    stats.pper.sbc,stats.pper.sem,...
    stats.xper.sbc,stats.xper.sem,...
    stats.yper.sbc,stats.yper.sem,...
    stats.areaper.sbc,stats.areaper.sem);
title('NC predicts person ratings');
%%
% get noise correlations for leave 1 subject out analysis
load L2_per.mat;load L2_car;
load L2_car;
load('L2_cp.mat');
id1024=L2_car.id1024(651:1290);
idabs=651:1300;
[c,stats.pcar.sem]=onevsrestcorr(L2_car.pcarraw(:,idabs),100);
stats.pcar.sbc=nanmean(c);
[c,stats.xcar.sem]=onevsrestcorr(squeeze(L2_cp.rectcar(id1024,:,1))',100);
stats.xcar.sbc=nanmean(c);
[c,stats.ycar.sem]=onevsrestcorr(squeeze(L2_cp.rectcar(id1024,:,2))',100);
stats.ycar.sbc=nanmean(c);
[c,stats.areacar.sem]=onevsrestcorr(squeeze(L2_cp.rectcar(id1024,:,1).*L2_cp.rectcar(id1024,:,2))',100);
stats.areacar.sbc=nanmean(c);
[c,stats.aspcar.sem]=onevsrestcorr(squeeze(L2_cp.rectcar(id1024,:,1)./L2_cp.rectcar(id1024,:,2))',100);
stats.aspcar.sbc=nanmean(c);

[c,stats.pper.sem]=onevsrestcorr(L2_car.ppersonraw(:,idabs),100);
stats.pper.sbc=nanmean(c);
[c,stats.xper.sem]=onevsrestcorr(squeeze(L2_cp.rectperson(id1024,:,1))',100);
stats.xper.sbc=nanmean(c);
[c,stats.yper.sem]=onevsrestcorr(squeeze(L2_cp.rectperson(id1024,:,2))',100);
stats.yper.sbc=nanmean(c);
[c,stats.areaper.sem]=onevsrestcorr(squeeze(L2_cp.rectperson(id1024,:,1).*L2_cp.rectperson(id1024,:,2))',100);
stats.areaper.sbc=nanmean(c);
[c,stats.aspper.sem]=onevsrestcorr(squeeze(L2_cp.rectperson(id1024,:,1)./L2_cp.rectperson(id1024,:,2))',100);
stats.aspper.sbc=nanmean(c);

fprintf('noise ceilings pcar %f(%f), xcar %f(%f), ycar %f(%f), areacar %f(%f) aspcar%f(%f)\n',...
    stats.pcar.sbc,stats.pcar.sem,...
    stats.xcar.sbc,stats.xcar.sem,...
    stats.ycar.sbc,stats.ycar.sem,...
    stats.areacar.sbc,stats.areacar.sem,...
    stats.aspcar.sbc,stats.aspcar.sem);
fprintf('noise ceilings pper %f(%f), xper %f(%f), yper %f(%f), areaper %f(%f) aspper %f(%f)\n',...
    stats.pper.sbc,stats.pper.sem,...
    stats.xper.sbc,stats.xper.sem,...
    stats.yper.sbc,stats.yper.sem,...
    stats.areaper.sbc,stats.areaper.sem,...
    stats.aspper.sbc,stats.aspper.sem);