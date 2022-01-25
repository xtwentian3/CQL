import gym
import d4rl

def test_env():
    env = gym.make("hopper-medium-v2")  # 'Walker2d-v2'  'HalfCheetah-v2'
    print(env.action_space)
    print(env.observation_space)
    _ = env.reset()
    for _ in range(100000):
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())
        # if done:
        #     break
    env.close()

test_env()