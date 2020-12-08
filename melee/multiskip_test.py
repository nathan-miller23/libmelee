from melee import SSBMEnv

ssbm_env = SSBMEnv(dolphin_exe_path="/Applications/dolphin-emu.app/Contents/MacOS", ssbm_iso_path="/Users/nathan/games/melee/SSMB.iso", cpu=False, every_nth=(4, 5), buffer_size=(2, 4))

obs = ssbm_env.reset()

assert 'ai_1' in obs and 'ai_2' in obs

done = False
i = -1
cycle = 8
while not done:
    joint_action = {}
    for agent in ['ai_1', 'ai_2']:
        if agent in obs:
            joint_action[agent] = ssbm_env.action_space.sample()
    assert joint_action

    if i % cycle == 0 or i == -1:
        assert 'ai_1' in obs and 'ai_2' in obs
    elif i % 2 == 0:
        assert 'ai_2' in obs and not 'ai_1' in obs
    else:
        assert 'ai_1' in obs and not 'ai_2' in obs
    obs, reward, done, info = ssbm_env.step(joint_action)
    i += 1
    assert obs.keys() == reward.keys()
    assert reward.keys() == info.keys()

    if 'ai_1' in obs:
        assert obs['ai_1'].shape[1] == 2
    if 'ai_2' in obs:
        assert obs['ai_2'].shape[1] == 4

    done = done['__all__']
