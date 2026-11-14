from pathlib import Path
import pandas as pd, torch
from lightning.pytorch import Trainer
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
R=Path(__file__).resolve().parents[1]; D=R/'data/synthetic_serverless_workload.csv'; O=R/'outputs'; df=pd.read_csv(D,parse_dates=['timestamp']); df['time_idx']=range(len(df)); df['series']=0; cutoff=df.time_idx.max()-48
train=TimeSeriesDataSet(df[df.time_idx<=cutoff],time_idx='time_idx',target='invocation_rate',group_ids=['series'],min_encoder_length=36,max_encoder_length=36,min_prediction_length=12,max_prediction_length=12,time_varying_known_reals=['time_idx','hour','day_of_week'],time_varying_unknown_reals=['invocation_rate','cpu_utilization','memory_utilization','latency_ms'],target_normalizer='auto',add_relative_time_idx=True,add_target_scales=True,add_encoder_length=True)
val=TimeSeriesDataSet.from_dataset(train,df,predict=True,stop_randomization=True); model=TemporalFusionTransformer.from_dataset(train,learning_rate=.03,hidden_size=16,attention_head_size=2,dropout=.1,hidden_continuous_size=8,output_size=7,loss=QuantileLoss()); Trainer(max_epochs=5,accelerator='auto',devices=1,logger=False,enable_checkpointing=False).fit(model,train.to_dataloader(train=True,batch_size=64,num_workers=0),val.to_dataloader(train=False,batch_size=64,num_workers=0)); O.mkdir(exist_ok=True); torch.save(model.state_dict(),O/'tft_state_dict.pt'); print('TFT training finished')
