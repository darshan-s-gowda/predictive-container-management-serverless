import sys, pandas as pd, numpy as np
sys.path.insert(0,'src')

df = pd.read_csv('outputs/end_to_end_results.csv')

print('=== 7-POLICY CALIBRATED COMPARISON (Azure test segment, simulated proxies) ===')
for policy in sorted(df['policy'].unique()):
    sub = df[df['policy']==policy]
    ml  = sub['mean_latency_ms'].mean()
    mls = sub['mean_latency_ms'].std()
    p95 = sub['p95_latency_ms'].mean()
    p9s = sub['p95_latency_ms'].std()
    sla = sub['sla_violation_rate'].mean()
    ss  = sub['sla_violation_rate'].std()
    wc  = sub['avg_warm_containers'].mean()
    wcs = sub['avg_warm_containers'].std()
    en  = sub['avg_energy_proxy'].mean()
    ens = sub['avg_energy_proxy'].std()
    print()
    print(policy + ':')
    print('  mean_latency_ms:      %.2f +/- %.2f' % (ml, mls))
    print('  p95_latency_ms:       %.2f +/- %.2f' % (p95, p9s))
    print('  sla_violation_rate:   %.4f +/- %.4f' % (sla, ss))
    print('  avg_warm_containers:  %.3f +/- %.3f' % (wc, wcs))
    print('  avg_energy_proxy:     %.4f +/- %.4f' % (en, ens))

ppo = df[df['policy']=='PPO (calibrated)']
if len(ppo):
    print()
    print('PPO action distribution (mean across seeds):')
    for col in ['action_noop','action_prewarm','action_scale_out','action_scale_in','action_terminate','action_reuse']:
        print('  %s: %.1f' % (col, ppo[col].mean()))
