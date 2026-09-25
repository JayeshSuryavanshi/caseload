# Superseded: the 320-episode single-seed run

These four files come from one PPO seed trained for 40 updates x 8 episodes = 320
episodes (`ppo_train.log`, `ppo_log.json`) and its evaluation (`evaluate.log`,
`evaluation.json`). The conclusion drawn from them was wrong and was retracted in
commit 8f28855. The current results are the six-seed, 1,600-episode fleet in
`results/fleet/`.

They are kept so the correction can be checked against what it corrects. Do not cite them.
