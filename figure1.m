% pick representative figure to show high likelihood car and high
% likelihood person scenes.
close all;
clc;
idabs=651:1300;
idtarget=1:650;

if ~exist('carstim')
    load cardetstim.mat;
    carstim.stim=stim;carstim.stimname=stimname;carstim.target=target;
end
if ~exist('perstim')
    load perdetstim.mat;
    perstim.stim=stim;perstim.stimname=stimname;perstim.target=target;
end
close all;
load L2_per.mat;load L2_car.mat;
[~,hc]=sort(L2_car.pcar(idabs),'descend');
[~,hp]=sort(L2_per.pperson(idabs),'descend');
%%
f=fspecial('gaussian',[50 50],15);
id=hc(6)+650;
imwrite(carstim.stim{id},'figure1_hc.bmp','bmp');
imwrite(imresize(imfilter(carstim.stim{id},f,'same'),0.5,'bilinear'),'figure1_hc_cntxt.bmp','bmp');
%%
id=hp(7)+650;
imwrite(perstim.stim{id},'figure1_hp.bmp','bmp');
imwrite(imresize(imfilter(perstim.stim{id},f,'same'),0.5,'bilinear'),'figure1_hp_cntxt.bmp','bmp');