%% Junhui Li, Liangdong Guo and Youshan Zhang
%SCORN: Sinter Composition Optimization with Regressive Convolutional Neural Network. 
% Solids. 2022; 3(3):416-429. https://doi.org/10.3390/solids3030029
clear;clc
load SCORNDATA
rng(999);

% x_test=reshape(testX',[1,1,1,size(testX,1)]); % input is production
% y_test=testY; % outputs are different indexes

armse=[];
mae=[];
indices=crossvalind('KFold',6242,5);
%% five-fold validation
for i=1:5
    test=(indices==i);
    train=~test;
    trainData=data(train,:);
    validationData=data(test,:);
    x_train=reshape(trainData(:,end)',[1,1,1,size(trainData(:,end),1)]); % input is production
    y_train= trainData(:,1:9); % outputs are different indexes

    x_validation=reshape(validationData(:,end)',[1,1,1,size(validationData(:,end),1)]); % input is production
    y_validation= validationData(:,1:9); % outputs are different indexes
  layers= [ 
           imageInputLayer([1 1 1]);
           convolution2dLayer(1,12);
           reluLayer
           batchNormalizationLayer
           averagePooling2dLayer(1,'Stride',2)
           crossChannelNormalizationLayer(2)
           maxPooling2dLayer(1,'Stride',2);
% 
           dropoutLayer(0.7);
           fullyConnectedLayer(size(y_train,2));
%           reluLayer
           regressionLayer];
% %   validInputSize=[1 1 1 6242]
%  checkLayer(layer,validInputSize,'ObservationDimension' )
%0.0001
%'Plots','training-progress'
options = trainingOptions('sgdm',...
                          'MaxEpochs',300,...
                          'InitialLearnRate',0.0005);
 %                     'Plots','training-progress'
 %                        
%'ValidationData',{x_test,y_test}
 [net, info] = trainNetwork(x_train,y_train,layers,options); 
%% Statistic 
 YPredicted = predict(net,x_validation);
 %ARMSE 
 
 sss=0;
  for j=1:size(y_validation,2)
     bb=0;
     for k=1:size(y_validation,1)
     bb=bb+double(y_validation(k,j)-YPredicted(k,j)).^2;
     end
     sss=sss+sqrt((bb)/size(y_validation,1));
  end
 armse(i)=sss/size(y_validation,2);
 
rmse(i)=mean(sqrt(mean((y_validation'-YPredicted').^2)));
mae(i)=mean(mean(abs(y_validation'-YPredicted')));
% Cacluate R-squared R2 
r =double( y_validation'-YPredicted');
normr =double(norm(r));
SSE = double(normr.^2);
SST = norm(double(y_validation'-mean(y_validation')))^2;
R2(i) = double(1 - SSE/SST);
end

disp(['RMSE is: ',num2str(mean(armse))])

disp(['MAE is: ',num2str(mean(mae))])

disp(['R2 is: ',num2str(mean(R2))])


