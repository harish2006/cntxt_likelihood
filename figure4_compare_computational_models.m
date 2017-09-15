% draw a single bar plot that compares performance of NC, gist, vgg, places
% for eachof 2 x 4 regressors
close all;
% get noise ceiling
% compute error bars, using only L2_cp for the car and person boxes
load L2_car;
load('L2_cp.mat');
id1024=L2_car.id1024(651:1290);
% also put in the noise ceiling on the training model
idabs=651:1300;
[c,~,~,stats.pcar.sem]=splithalfcorr(L2_car.pcarraw(:,idabs),100);
stats.pcar.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.xcar.sem]=splithalfcorr(squeeze(L2_cp.rectcar(id1024,:,1))',100);
stats.xcar.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.ycar.sem]=splithalfcorr(squeeze(L2_cp.rectcar(id1024,:,2))',100);
stats.ycar.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.areacar.sem]=splithalfcorr(squeeze(L2_cp.rectcar(id1024,:,1).*L2_cp.rectcar(id1024,:,2))',100);
stats.areacar.sbc=spearmanbrowncorrection(c,2);
xx=squeeze(L2_cp.rectcar(id1024,:,3)./L2_cp.rectcar(id1024,:,4))';
xx(:,find(isnan(nanmean(xx,1))))=[];
[c,~,~,stats.aspcar.sem]=mysplithalfcorr(xx,100);
stats.aspcar.sbc=spearmanbrowncorrection(c,2);

[c,~,~,stats.pper.sem]=splithalfcorr(L2_car.ppersonraw(:,idabs),100);
stats.pper.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.xper.sem]=splithalfcorr(squeeze(L2_cp.rectperson(id1024,:,1))',100);
stats.xper.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.yper.sem]=splithalfcorr(squeeze(L2_cp.rectperson(id1024,:,2))',100);
stats.yper.sbc=spearmanbrowncorrection(c,2);
[c,~,~,stats.areaper.sem]=splithalfcorr(squeeze(L2_cp.rectperson(id1024,:,1).*L2_cp.rectperson(id1024,:,2))',100);
stats.areaper.sbc=spearmanbrowncorrection(c,2);
xx=squeeze(L2_cp.rectperson(id1024,:,3)./L2_cp.rectperson(id1024,:,4))';
[c,~,~,stats.aspper.sem]=mysplithalfcorr(xx,100);
stats.aspper.sbc=spearmanbrowncorrection(c,6);

load rvals_likelihood_100_wgist.mat;rvals_nc=rvals;
% load rvals_likelihood_100_gist.mat;rvals_gist=rvals;
rvals_gist=rvals;
load rvals_likelihood_100_fullVgg;rvals_v=rvals;
load rvals_likelihood_100_fullPlaces205;rvals_p=rvals;
load rvals_likelihood_100_pixels;rvals_pix=rvals;

vals.pcar.c=[nanmean(rvals_pix.pcar(2,:),2) nanmean(rvals_nc.pcar(7,:),2) nanmean(rvals_gist.pcar(2,:),2) nanmean(rvals_v.pcar(2,:),2) nanmean(rvals_p.pcar(2,:),2)];
vals.pcar.e=[nanstd(rvals_pix.pcar(2,:),[],2) nanstd(rvals_nc.pcar(7,:),[],2) nanstd(rvals_gist.pcar(2,:),[],2) nanstd(rvals_v.pcar(2,:),[],2) nanstd(rvals_p.pcar(2,:),[],2)];

vals.xcar.c=[nanmean(rvals_pix.xcar(2,:),2) nanmean(rvals_nc.xcar(7,:),2) nanmean(rvals_gist.xcar(2,:),2) nanmean(rvals_v.xcar(2,:),2) nanmean(rvals_p.xcar(2,:),2)];
vals.xcar.e=[nanstd(rvals_pix.xcar(2,:),[],2) nanstd(rvals_nc.xcar(7,:),[],2) nanstd(rvals_gist.xcar(2,:),[],2) nanstd(rvals_v.xcar(2,:),[],2) nanstd(rvals_p.xcar(2,:),[],2)];

vals.ycar.c=[nanmean(rvals_pix.ycar(2,:),2) nanmean(rvals_nc.ycar(7,:),2) nanmean(rvals_gist.ycar(2,:),2) nanmean(rvals_v.ycar(2,:),2) nanmean(rvals_p.ycar(2,:),2)];
vals.ycar.e=[nanstd(rvals_pix.ycar(2,:),[],2) nanstd(rvals_nc.ycar(7,:),[],2) nanstd(rvals_gist.xcar(2,:),[],2) nanstd(rvals_v.ycar(2,:),[],2) nanstd(rvals_p.ycar(2,:),[],2)];

vals.areacar.c=[nanmean(rvals_pix.areacar(2,:),2) nanmean(rvals_nc.areacar(7,:),2) nanmean(rvals_gist.areacar(2,:),2) nanmean(rvals_v.areacar(2,:),2) nanmean(rvals_p.areacar(2,:),2)];
vals.areacar.e=[nanstd(rvals_pix.areacar(2,:),[],2) nanstd(rvals_nc.areacar(7,:),[],2) nanstd(rvals_gist.areacar(2,:),[],2) nanstd(rvals_v.areacar(2,:),[],2) nanstd(rvals_p.areacar(2,:),[],2)];

vals.aspcar.c=[nanmean(rvals_pix.aspcar(2,:),2) nanmean(rvals_nc.aspcar(7,:),2) nanmean(rvals_gist.aspcar(2,:),2) nanmean(rvals_v.aspcar(2,:),2) nanmean(rvals_p.aspcar(2,:),2)];
vals.aspcar.e=[nanstd(rvals_pix.aspcar(2,:),[],2) nanstd(rvals_nc.aspcar(7,:),[],2) nanstd(rvals_gist.aspcar(2,:),[],2) nanstd(rvals_v.aspcar(2,:),[],2) nanstd(rvals_p.aspcar(2,:),[],2)];

%% for people
vals.pper.c=[nanmean(rvals_pix.pper(2,:),2) nanmean(rvals_nc.pper(7,:),2) nanmean(rvals_gist.pper(2,:),2) nanmean(rvals_v.pper(2,:),2) nanmean(rvals_p.pper(2,:),2)];
vals.pper.e=[nanstd(rvals_pix.pper(2,:),[],2) nanstd(rvals_nc.pper(7,:),[],2) nanstd(rvals_gist.pper(2,:),[],2) nanstd(rvals_v.pper(2,:),[],2) nanstd(rvals_p.pper(2,:),[],2)];

vals.xper.c=[nanmean(rvals_pix.xper(2,:),2) nanmean(rvals_nc.xper(7,:),2) nanmean(rvals_gist.xper(2,:),2) nanmean(rvals_v.xper(2,:),2) nanmean(rvals_p.xper(2,:),2)];
vals.xper.e=[nanstd(rvals_pix.xper(2,:),[],2) nanstd(rvals_nc.xper(7,:),[],2) nanstd(rvals_gist.xper(2,:),[],2) nanstd(rvals_v.xper(2,:),[],2) nanstd(rvals_p.xper(2,:),[],2)];

vals.yper.c=[nanmean(rvals_pix.yper(2,:),2) nanmean(rvals_nc.yper(7,:),2) nanmean(rvals_gist.yper(2,:),2) nanmean(rvals_v.yper(2,:),2) nanmean(rvals_p.yper(2,:),2)];
vals.yper.e=[nanstd(rvals_pix.yper(2,:),[],2) nanstd(rvals_nc.yper(7,:),[],2) nanstd(rvals_gist.xper(2,:),[],2) nanstd(rvals_v.yper(2,:),[],2) nanstd(rvals_p.yper(2,:),[],2)];

vals.areaper.c=[nanmean(rvals_pix.areaper(2,:),2) nanmean(rvals_nc.areaper(7,:),2) nanmean(rvals_gist.areaper(2,:),2) nanmean(rvals_v.areaper(2,:),2) nanmean(rvals_p.areaper(2,:),2)];
vals.areaper.e=[nanstd(rvals_pix.areaper(2,:),[],2) nanstd(rvals_nc.areaper(7,:),[],2) nanstd(rvals_gist.areaper(2,:),[],2) nanstd(rvals_v.areaper(2,:),[],2) nanstd(rvals_p.areaper(2,:),[],2)];

vals.aspper.c=[nanmean(rvals_pix.aspper(2,:),2) nanmean(rvals_nc.aspper(7,:),2) nanmean(rvals_gist.aspper(2,:),2) nanmean(rvals_v.aspper(2,:),2) nanmean(rvals_p.aspper(2,:),2)];
vals.aspper.e=[nanstd(rvals_pix.aspper(2,:),[],2) nanstd(rvals_nc.aspper(7,:),[],2) nanstd(rvals_gist.aspper(2,:),[],2) nanstd(rvals_v.aspper(2,:),[],2) nanstd(rvals_p.aspper(2,:),[],2)];


%% now plot everything
figure(101);
subplot(121);
barwitherr([vals.pper.e([1 2 4 5]);vals.xper.e([1 2 4 5]);vals.yper.e([1 2 4 5]);vals.areaper.e([1 2 4 5]);vals.aspper.e([1 2 4 5])],...
    [vals.pper.c([1 2 4 5]);vals.xper.c([1 2 4 5]);vals.yper.c([1 2 4 5]);vals.areaper.c([1 2 4 5]);vals.aspper.c([1 2 4 5])]);
hold on;
errorbar([1 2 3 4 5],...
    [stats.pper.sbc stats.xper.sbc stats.yper.sbc stats.areaper.sbc stats.aspper.sbc],...
    [stats.pper.sem stats.xper.sem stats.yper.sem stats.areaper.sem stats.aspper.sem],'.');
set(gca,'XTickLabel',{'Likelihood','Horizontal','Vertical','Area','Asp'});
% legend('pix','NC','obj-cnn','scene-cnn');
ylabel('Model correlation');
title('Person ratings');
ylim([0 1]);
subplot(122);
barwitherr([vals.pcar.e([1 2 4 5]);vals.xcar.e([1 2 4 5]);vals.ycar.e([1 2 4 5]);vals.areacar.e([1 2 4 5]);vals.aspcar.e([1 2 4 5])],...
    [vals.pcar.c([1 2 4 5]);vals.xcar.c([1 2 4 5]);vals.ycar.c([1 2 4 5]);vals.areacar.c([1 2 4 5]);vals.aspcar.c([1 2 4 5])]);
hold on;
errorbar([1 2 3 4 5],...
    [stats.pcar.sbc stats.xcar.sbc stats.ycar.sbc stats.areacar.sbc stats.aspcar.sbc],...
    [stats.pcar.sem stats.xcar.sem stats.ycar.sem stats.areacar.sem stats.aspcar.sem],'.'); % standard deviation of noise ceiling
set(gca,'XTickLabel',{'Likelihood','Horizontal','Vertical','Area','Asp'});
% legend('pix','NC','obj-cnn','scene-cnn');
ylabel('Model correlation');
title('Car ratings');
ylim([0 1]);