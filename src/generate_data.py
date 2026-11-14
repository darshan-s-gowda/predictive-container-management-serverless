from pathlib import Path
import numpy as np, pandas as pd
R=Path(__file__).resolve().parents[1]; OUT=R/'data/synthetic_serverless_workload.csv'
rng=np.random.default_rng(42); n=14*24*12; ts=pd.date_range('2026-09-01',periods=n,freq='5min'); i=np.arange(n); hour=(i/12)%24; dow=(i//288)%7
daily=18+14*np.maximum(0,np.sin((hour-7)*np.pi/12)); weekly=np.where(dow<5,1,.65); bursts=sum(35*np.exp(-.5*((i-c)/25)**2) for c in [700,1200,2100,3000]); rate=np.maximum(2,daily*weekly+bursts+rng.normal(0,3,n))
cpu=np.clip(15+rate*1.7+rng.normal(0,5,n),5,98); mem=np.clip(22+rate*.9+rng.normal(0,4,n),10,95); lat=np.clip(75+np.maximum(rate-35,0)*5+rng.normal(0,8,n),35,900)
df=pd.DataFrame({'timestamp':ts,'invocation_rate':rate.round(3),'cpu_utilization':cpu.round(3),'memory_utilization':mem.round(3),'latency_ms':lat.round(3)}); df['hour']=df.timestamp.dt.hour; df['day_of_week']=df.timestamp.dt.dayofweek; OUT.parent.mkdir(exist_ok=True); df.to_csv(OUT,index=False); print('Wrote',OUT)
if __name__=='__main__': pass
