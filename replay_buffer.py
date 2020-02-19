#!/usr/bin/env python
import sys
import numpy as np

import zlib
import collections
class ReplayBuffer(object):
    def __init__(self, batch_size,
                 state_history_size=4,
                 max_buffer_size=1E6,
                 garbage_check_multiplier=1.0,
                 random_state=None):
        super(ReplayBuffer, self).__init__()
        # stores "pointers" into the reverse experience lookup
        self.memory = []
        # stores zipped versions of states, indexed by sent_counter step
        self.reverse_experience_lookup = collections.defaultdict(dict)
        self.batch_size = batch_size
        self.sent_counter = 0
        self.max_buffer_size = max_buffer_size
        self.garbage_check_multiplier=garbage_check_multiplier
        self.state_history_size = state_history_size
        self.state_info = None
        if random_state is None:
            self.random_state = np.random.RandomState(1234)
        else:
            self.random_state = random_state

    def add_experience(self, partial_experience):
        # partial experience of form

        # partial_experience = (S_prime, action, reward, ..., terminal_flag)
        # a tuple with transitioned-to state in the 0th entry
        # all other info afterwards
        self.state_info = {}
        self.state_info["shape"] = partial_experience[0].shape
        self.state_info["dtype"] = partial_experience[0].dtype

        S_prime_bytes = partial_experience[0].tostring()
        z_S_prime = zlib.compress(S_prime_bytes)

        # all of this compression, lack of copies is to minimize memory
        # THIS LOGIC ASSUMES EPISODE STEPS ARE SENT IN ORDER

        if self.sent_counter not in self.reverse_experience_lookup:
            self.reverse_experience_lookup[self.sent_counter]["S_prime"] = z_S_prime
        else:
            print("ERROR: step already found in exp lookup?")
            from IPython import embed; embed(); raise ValueError()

        self.memory.append((self.sent_counter,) + partial_experience[1:])
        self.sent_counter += 1

        # garbage collection / cleanup here
        # + extra to account for the offset of S / S_prime in the lookup
        # garbage check so that we don't do this garbage collection too often
        if len(self.memory) > (self.garbage_check_multiplier * self.max_buffer_size + 1):
            while len(self.memory) > (self.max_buffer_size + 1):
                cleanup = self.memory.pop(0)
                del self.reverse_experience_lookup[cleanup[0]]

    def get_minibatch(self, index_list=None):
        if len(self.memory) > self.batch_size + self.state_history_size + 1:
            # if this is n_step returns we will need more trickery
            if index_list is None:
                draw = self.random_state.choice(len(self.memory) - self.state_history_size - 1,
                                                size=self.batch_size,
                                                replace=False)
                return_experience = [self.memory[d + self.state_history_size + 1] for d in draw]
            else:
                draw = index_list
                # todo: some kind of warning here?
                return_experience = [self.memory[d] for d in draw]

            b = []
            for r_e in return_experience:
                # example format for rev exp lookup
                """
                defaultdict(dict,
                {0: {'S_prime': 'x\x9c\xed\xd4\xb1\r\x800\x0cDQW\x1e\x81\x11\x18!#0:s\xd1R\x12\xe9,r\xce\xff\x8d\xe5\xe6U\x96#\xa8C\xf9\xea\xcb\xfe\xa7931\xfb\x98\x15\xb7\xa46W(\x89\x886\xaa\xe2\x7fbbbbbbb\xf63/a\x98>\xe6\x10\x86\xe9c\x9e\xc20}\xccC\x18\xa6\x8fy\x0bs1\x1f\x16A&?'}})
                """
                primary_idx = r_e[0]
                if primary_idx > self.state_history_size:
                    all_idx = list(range(primary_idx - self.state_history_size, primary_idx + 1))
                else:
                    all_idx = [a if a > self.state_history_size
                                 else -1
                                 for a in list(range(primary_idx - self.state_history_size, primary_idx + 1))]

                    # handle negative indices index_list requests < self.state_history_size here by making fake zeros
                    # may need to do similar if request crosses terminal state
                    zero_bytes = np.zeros(self.state_info["shape"], dtype=self.state_info["dtype"]).tostring()
                    z_zero_bytes = zlib.compress(zero_bytes)

                # deepcopy to avoid references
                _all_zlib_S = [self.reverse_experience_lookup[i]["S_prime"] if i >= 0 else z_zero_bytes for i in all_idx]
                # some shared computation here, could cache the decompressed ones
                def undo(z):
                    return np.frombuffer(zlib.decompress(z),
                                         dtype=self.state_info["dtype"]).reshape(self.state_info["shape"])

                _all_S = [undo(z) for z in _all_zlib_S]

                # decompress
                # standard mode is N-step = 1
                b_i = (np.concatenate((np.stack(_all_S[:-1]).transpose(1, 2, 0)[None], np.stack(_all_S[1:]).transpose(1, 2, 0)[None]))[None],) + r_e[1:]
                b.append(b_i)
            # make a proper minibatch here?
            # current format is list of tuple
            S = np.concatenate([bi[0] for bi in b])
            stacked_tuple_info = [bi[1:] for bi in b]
            return S, stacked_tuple_info
        else:
           return None


if __name__ == "__main__":
    from atari_py.ale_python_interface import ALEInterface
    import os

    if len(sys.argv) < 2:
        print('Using default game atari_roms/breakout.bin')
        game = "atari_roms/breakout.bin"
    else:
        game = sys.argv[1]

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
            #if batch is not None:
            #    mb_S = batch[0]
            #    mb_S_prime = batch[1]
            #    other_info = batch[2]
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
        mb_S_prime = batch[1]
        other_info = batch[2]
    from IPython import embed; embed(); raise ValueError()
