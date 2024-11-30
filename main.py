from envs import GridWorldEnv
import matplotlib.pyplot as plt
from torchrl.envs import GymWrapper


env = GridWorldEnv(render_mode="rgb_array", size=10)
env.reset()
img1 = env.render()
plt.imshow(img1)
plt.title(label='GridWorldEnv: OpenAI Gym')
plt.show()

trl_env = GymWrapper(env, from_pixels=True, pixels_only=False)
td_env = trl_env.reset()
img2 = td_env['pixels']
plt.imshow(img2)
plt.title(label='GridWorldEnv: TorchRL GymWrapper')
plt.show()