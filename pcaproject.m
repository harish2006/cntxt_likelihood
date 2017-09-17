% data = nobs x nfeatures

function [proj,npc] = pcaproject(data,pcthresh)
if(~exist('pcthresh')), pcthresh = 0.99; end; 

[pc,proj,ev] = princomp(data);
cumvar = cumsum(ev/sum(ev)); % cumulative percent variance explained
npc = min(find(cumvar>pcthresh));
proj = proj(:,1:npc);

return
