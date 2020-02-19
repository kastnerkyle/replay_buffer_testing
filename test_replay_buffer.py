from replay_buffer import ReplayBuffer
import numpy as np

def test_full_run():
    from atari_py.ale_python_interface import ALEInterface

    game = "atari_roms/breakout.bin"

    ale = ALEInterface()

    # Get & Set the desired settings
    ale.setInt('random_seed', 123)

    # Load the ROM file
    ale.loadROM(game)

    # Get the list of legal actions
    legal_actions = ale.getLegalActionSet()

    batch_size = 10
    exp_replay = ReplayBuffer(batch_size)

    (screen_width, screen_height) = ale.getScreenDims()

    import os
    tot_m, used_m, free_m = os.popen("free -th").readlines()[-1].split()[1:]
    last_counter = 0
    random_state = np.random.RandomState(218)
    print("initial: {}, {}, {}".format(tot_m, used_m, free_m))
    # Play 2k episodes
    for episode in range(2000):
        total_reward = 0
        S = np.zeros(screen_width*screen_height, dtype=np.uint8)
        S = S.reshape(screen_height, screen_width)[:84, :84]
        this_counter = exp_replay.sent_counter
        if this_counter > last_counter + 1000:
            last_counter = this_counter
            tot_m, used_m, free_m = os.popen("free -th").readlines()[-1].split()[1:]
            # the first three entries should match til 1M steps
            # then the second 2 should continue in lock step
            print("{}: {}, {}; {}, {}, {}".format(exp_replay.sent_counter, len(exp_replay.memory), len(exp_replay.reverse_experience_lookup.keys()), tot_m, used_m, free_m))
        while not ale.game_over():
            S_prime = np.zeros(screen_width*screen_height, dtype=np.uint8)
            ale.getScreen(S_prime)
            S_prime = S_prime.reshape(screen_height, screen_width)[:84, :84]
            a = random_state.choice(len(legal_actions))
            action = legal_actions[a]
            # Apply an action and get the resulting reward
            reward = ale.act(action)
            won = 0
            ongoing_flag = 1
            experience = (S_prime, action, reward, won, ongoing_flag)
            S = S_prime
            exp_replay.add_experience(experience)
            batch = exp_replay.get_minibatch()
            batch = exp_replay.get_minibatch(index_list=[1, 2, 3, 10, 11])
            if batch is not None:
                mb_S = batch[0]
                other_info = batch[1]
            del batch
            total_reward += reward
        print 'Episode', episode, 'ended with score:', total_reward
        ale.reset_game()

    lst = 0
    for i in range(10000):
        if i > lst + 1000:
            tot_m, used_m, free_m = os.popen("free -th").readlines()[-1].split()[1:]
            print("POST MEM {}: {}, {}; {}, {}, {}".format(exp_replay.sent_counter, len(exp_replay.memory), len(exp_replay.reverse_experience_lookup.keys()), tot_m, used_m, free_m))
            lst = i

        batch = exp_replay.get_minibatch()
        mb_S = batch[0]
        other_info = batch[1]
    from IPython import embed; embed(); raise ValueError()


test_full_run()
