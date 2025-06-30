env='SimplePacman-v0' # Possible environments: ['SimplePacman-v0', 'DungeonCrawler-v0', 'SimpleMinecraft-v0']
env_num=1             # index of the environment (based on the order above)
env_map="map.txt"     # REQUIRED FOR PACMAN AND DUNGEON CRAWLER ONLY. Remove for MineCraft.
timesteps=1000000
for seed in 12 23 34 43 54 65 76 87 98 109; do
    # 1-[1-3].0
    exp_id="1-$env_num.0"
    echo $env $env_map $exp_id
    python main.py --exp_id "${exp_id}" --env "${env}" --env_map "${env_map}" --timesteps "${timesteps}" --seed "${seed}"

    # 1-[1-3].[1-3]
    for shaping in 1 2 3; do
        exp_id="1-$env_num.$shaping"
        echo $env $env_map $shaping $exp_id
        python main.py --exp_id "${exp_id}" --shaping "${shaping}" --env "${env}" --env_map "${env_map}" --timesteps "${timesteps}" --seed "${seed}"
    done

    # 1-[1-3].1.[1-3]
    j=1
    shaping=1
    for decay_param in 0.99 0.95 0.9; do
        exp_id="1-$env_num.$shaping.$j"
        echo $env $env_map $shaping $exp_id $decay_param
        python main.py --exp_id "${exp_id}" --shaping "${shaping}" --env "${env}" --env_map "${env_map}" --timesteps "${timesteps}" -d --decay_param "${decay_param}" --decay_n 0 --seed "${seed}"
        ((j++))
    done
done
