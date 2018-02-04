# Wind-Prediction-RNN
Vanilla RNN is used to predict the wind energy from wind farm over the next time slots. 
Number of LSTM cells is kept as 32. This is based on th etime series model that produced good AIC value for SARIMA (4,0,0)x(1,0,0) model.

The predicted wind energy from this model will be used for optimization where we make decisions at time $t$, based on past information and future expected values. The model is as good as prediction model as if the model is bad, decisions at time $t$ will suffer. Thus, an ensemble method is used to predict wind energy in next steps. A sampling method is used to sample predictions and the performace is measured using Kantorovic distance.

It considers correlation betwen different data points which is important since wind generation follows time series patrern and i.i.d assumption does not hold. The samples are drawn such that the Kantorovic distance between the original data and the samples does not decrease any further.
