from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]; D=R/'data/synthetic_serverless_workload.csv'; O=R/'outputs'; O.mkdir(exist_ok=True)
if not D.exists():
 import subprocess,sys; subprocess.check_call([sys.executable,str(R/'src/generate_data.py')])
df=pd.read_csv(D,parse_dates=['timestamp']); recent=df.invocation_rate.tail(36).to_numpy(); coef=np.polyfit(np.arange(len(recent)),recent,1); pred=np.maximum(1,np.polyval(coef,np.arange(36,48))); times=pd.date_range(df.timestamp.iloc[-1]+pd.Timedelta(minutes=5),periods=12,freq='5min'); f=pd.DataFrame({'timestamp':times,'predicted_invocation_rate':pred}); f.to_csv(O/'workload_forecast.csv',index=False)
plt.figure(figsize=(10,5)); plt.plot(df.timestamp.tail(72),df.invocation_rate.tail(72),label='Observed workload'); plt.plot(f.timestamp,f.predicted_invocation_rate,'o-',label='Illustrative forecast'); plt.axvline(df.timestamp.iloc[-1],ls='--',label='Forecast start'); plt.xlabel('Time'); plt.ylabel('Invocation rate'); plt.title('Module 1 — Workload Prediction Demo (illustrative baseline)'); plt.legend(); plt.tight_layout(); plt.savefig(O/'workload_forecast.png',dpi=180); plt.close()
warm=4; rows=[]
for _,r in f.iterrows():
 demand=float(r.predicted_invocation_rate); required=int(np.ceil(demand/10))
 if required>warm+1: action='Scale out'; warm=max(warm+2,required)
 elif required>warm: action='Pre-warm'; warm+=1
 elif required<warm-2: action='Scale in'; warm=max(1,warm-1)
 else: action='Reuse warm container'
 lat=70+max(0,demand-warm*10)*8; energy=round(warm*.8+.015*min(100,15+demand*1.5),3); rows.append([r.timestamp,demand,warm,action,lat,energy,'VIOLATION' if lat>200 else 'OK'])
rdf=pd.DataFrame(rows,columns=['timestamp','predicted_invocation_rate','warm_containers','action','estimated_latency_ms','estimated_energy','sla_status']); rdf.to_csv(O/'ppo_demo_actions.csv',index=False)
fig,ax=plt.subplots(figsize=(10,5)); ax.plot(rdf.timestamp,rdf.predicted_invocation_rate,'o-',label='Predicted workload'); ax.set_ylabel('Invocation rate'); ax2=ax.twinx(); ax2.step(rdf.timestamp,rdf.warm_containers,where='mid',label='Warm containers'); ax2.set_ylabel('Warm containers'); ax.set_xlabel('Time'); plt.title('Module 2 — Container Management Demo (prototype policy)'); fig.tight_layout(); fig.savefig(O/'container_management.png',dpi=180); plt.close(fig); print('Created demo CSVs and plots in outputs/')
