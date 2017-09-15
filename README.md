# cntxt_likelihood
Each object in the world occurs in a specific context: cars are seen on highways but not in forests. Context nerally thought to facilitate computation by constraining locations to search. But can knowing context yield tangible benefits in object detection? For it to do so, scene context needs to be learned independently from target features. However this is impossible in traditional object detection where classifiers are trained on images containing both target features and surrounding coarse scene features. In contrast, we humans have the opportunity to learn context and target features separately, such as when we see highways without cars. Here we show for the first time that human-derived scene expectations can be used to improve object detection performance in machines. 
To measure contextual expectations in humans, we asked subjects to indicate the scale, location and likelihood at which cars or people may occur within scenes that contained neither of these objects. Humans showed highly systematic expectations that we could accurately predict using scene features. We then augmented state-of-the-art object detectors (based on deep neural networks) with these predicted human-derived expectations on novel scenes. This yielded a significant (1-3%) improvement in detecting cars and people in scenes and even on detecting associated objects. This improvement was due to relatively poor matches at highly likely locations being correctly labelled as target and conversely strong matches at unlikely locations being correctly rejcted as false alarms. Our results show that human-derived contextual features can and do improve state-of-the-art object detectors.

Preprint:https://arxiv.org/abs/1611.07218 This release has the visual features, images, behavioural data and analysis code to generate figures in the paper.

Images, extracted visual features and behavioural data from human experiments will be made available soon.

Please email harish2006@gmail.com for more details or queries.

All the code, data, images and visual features are made availalble purely for research purposes and should not be used for commercial gain.
