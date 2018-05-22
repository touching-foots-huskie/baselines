import gym
env = gym.make('Reacher-v2')
i = 0
env.reset()
while i < 100:
    env.step([-1.0, -1.0])
    env.render()
    i += 1
