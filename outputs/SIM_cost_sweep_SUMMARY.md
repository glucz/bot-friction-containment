# Cost-parameter sweep of the policy ranking (one-at-a-time from base)

| Setting | Binary L_D | Throttle L_D | Graded L_D | Throttle worst? | Graded best? | Best dashboard = throttle? | Graded/binary harm |
|---|---|---|---|---|---|---|---|
| c_E=0.25 | 98.9 | 105.3 | 92.9 | True | True | True | 0.092 |
| c_E=0.35 (base) | 91.5 | 96.4 | 84.8 | True | True | True | 0.479 |
| c_E=0.5 | 82.6 | 86.3 | 74.2 | True | True | True | 0.27 |
| rho=0.2 | 98.8 | 104.2 | 87.3 | True | True | True | 0.135 |
| rho=0.3 (base) | 91.5 | 96.4 | 84.8 | True | True | True | 0.479 |
| rho=0.45 | 82.7 | 87.0 | 77.1 | True | True | True | 0.692 |
| K_E=0.25 | 93.0 | 97.7 | 85.3 | True | True | True | 0.17 |
| K_E=0.5 (base) | 91.5 | 96.4 | 84.8 | True | True | True | 0.479 |
| K_E=1.0 | 89.9 | 94.9 | 84.1 | True | True | True | 0.541 |
| q=1.0 | 91.5 | 112.8 | 94.2 | True | False | True | 0.546 |
| q=1.5 | 91.5 | 102.6 | 88.1 | True | True | True | 0.275 |
| q=2.0 (base) | 91.5 | 96.4 | 84.8 | True | True | True | 0.479 |
| q=3.0 | 91.5 | 89.3 | 78.9 | False | True | True | 0.396 |