CUDA_VISIBLE_DEVICES=$1 nohup python -m dqn_zoo.$3.run_atari --environment_name=$2  \
--jax_platform_name=gpu --results_csv_path=./results/$3_25atoms_$2.csv --num_iterations=50 \
--num_atoms=25 \
  &> outputs/nohup_$3_25atoms_$2.out &
