# Path Consistency Learning in Autonomous Driving

This is an implementation of Path Consistency Learning in Autonomous Driving, using the end2end control.
The algorithms are from these papers:
"Improving Policy Gradient by Exploring Under-appreciated Rewards" by Ofir Nachum, Mohammad Norouzi, and Dale Schuurmans.
"Bridging the Gap Between Value and Policy Based Reinforcement Learning" by Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans.
"Trust-PCL: An Off-Policy Trust Region Method for Continuous Control" by Ofir Nachum, Mohammad Norouzi, Kelvin Xu, and Dale Schuurmans.
"Path Consistency Learning in Tsallis Entropy Regularized MDPs" by Ofir Nachum, Yinlam Chow, Mohamamd Ghavamzadeh.

The Path Consistency Learning code is based on the above work but in TF1. here is in TF2, which is also can do some experiments in OpenAL GYM.

The simluator used in here is CARLA. Version: 0.9.9.4. 

Quick start:

To run the experiment, first initialize the CARLA simluator in the CARLA directory:
```
./CarlaUE4.sh
```

And run the experiment with Trust-PCL here:
```
python trainer.py --logtostderr --batch_size=1 --env=carla \
  --validation_frequency=200 --rollout=5 --critic_weight=1.0 --gamma=0.995 \
  --clip_norm=20 --learning_rate=0.0001 \
  --replay_buffer_alpha=0.001 --norecurrent \
  --objective=trust_pcl --max_step=50 --cutoff_agent=400 --tau=0.20 --eviction=fifo \
  --max_divergence=0.001 --replay_batch_size=64 \
  --nouse_online_batch --batch_by_steps --value_hidden_layers=2 \
  --nounify_episodes --target_network_lag=0.99 --max_steer=0.8 --min_steer=-0.8\
  --clip_adv=1 --prioritize_by=step --num_steps=40000 --steer_dim=40 --throttle_dim=5\
  --noinput_prev_actions --use_target_values --tf_seed=37 --yaw_sig=0.2 --direction=0
```

Or to run the experiment with Sparse-PCL:
```
python trainer.py --logtostderr --batch_size=1 --env=carla \
  --validation_frequency=200 --rollout=5 --critic_weight=1.0 --gamma=0.995 \
  --clip_norm=20 --learning_rate=0.0001 \
  --replay_buffer_alpha=0.001 --norecurrent \
  --objective=trust_pcl --max_step=50 --cutoff_agent=400 --tau=0.20 --eviction=fifo \
  --max_divergence=0.001 --replay_batch_size=64 \
  --nouse_online_batch --batch_by_steps --value_hidden_layers=2 \
  --nounify_episodes --target_network_lag=0.99 --max_steer=0.8 --min_steer=-0.8\
  --clip_adv=1 --prioritize_by=step --num_steps=40000 --steer_dim=40 --throttle_dim=5\
  --noinput_prev_actions --use_target_values --tf_seed=37 --yaw_sig=0.2 --direction=0
```

Or to evaluate the trained model:
```
python trainer.py --logtostderr --notrain \
--path=./{saved model} --nums_record=0
```

Any further questions please contact hrddq512@126.com.