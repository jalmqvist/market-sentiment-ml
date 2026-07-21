| ID        | Finding                                                      | Evidence                             | Status      |
| --------- | ------------------------------------------------------------ | ------------------------------------ | ----------- |
| F-001     | Raw sentiment has negligible standalone predictive value     | V1–V29 validation                    | Confirmed   |
| F-002     | Behavioral State reproducibility survives leakage correction | Random-seed sweep 1.6.0              | Confirmed   |
| F-003     | LSTM consistently outperforms MLP on `JPY_CONSENSUS_MATURING` | Seed sweeps                          | Confirmed   |
| F-004     | MLP learns highly reproducible but weaker solutions          | Seed sweeps                          | Confirmed   |
| F-005     | Persistent LVTF transfer on 1.6.0 remains viable             | MPML benchmark                       | Preliminary |
| **F-006** | Reactive-JPY Behavioral Prediction Artifacts successfully <br />improve downstream MPML adaptive strategy selection | Initial MPML integration experiments | Preliminary |
| **F-007** | Reactive-JPY and Trend/Volatility Behavioral Surfaces produce substantially different prediction artifacts, indicating that independently validated Behavioral Surfaces encode distinct predictive representations rather than alternative parameterizations of a common predictive signal | Prediction Artifact Diversity Analysis | **Confirmed** |

