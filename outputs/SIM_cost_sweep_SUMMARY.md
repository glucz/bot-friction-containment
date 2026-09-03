# Cost-parameter sweep of the policy ranking (one-at-a-time from base)

| Setting | Binary L_D | Throttle L_D | Graded L_D | Throttle worst? | Graded best? | Best dashboard = throttle? | Graded/binary harm |
|---|---|---|---|---|---|---|---|
| c_E=0.25 | 98.9 | 106.5 | 94.8 | True | True | True | 0.749 |
| c_E=0.35 (base) | 91.5 | 97.1 | 86.3 | True | True | True | 0.728 |
| c_E=0.5 | 82.6 | 86.6 | 75.7 | True | True | True | 0.248 |
| rho=0.2 | 98.8 | 104.7 | 91.0 | True | True | True | 0.105 |
| rho=0.3 (base) | 91.5 | 97.1 | 86.3 | True | True | True | 0.728 |
| rho=0.45 | 82.7 | 87.6 | 78.0 | True | True | True | 0.728 |
| K_E=0.25 | 93.0 | 98.5 | 87.3 | True | True | True | 0.135 |
| K_E=0.5 (base) | 91.5 | 97.1 | 86.3 | True | True | True | 0.728 |
| K_E=1.0 | 89.9 | 95.6 | 85.0 | True | True | True | 0.728 |
| q=1.0 | 91.5 | 113.5 | 95.8 | True | False | True | 0.516 |
| q=1.5 | 91.5 | 103.3 | 89.9 | True | True | True | 0.252 |
| q=2.0 (base) | 91.5 | 97.1 | 86.3 | True | True | True | 0.728 |
| q=3.0 | 91.5 | 90.0 | 80.0 | False | True | True | 0.427 |