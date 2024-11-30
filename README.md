# Grid World Environment
An Example for creating environment using OpenAI Gym. Then this environment is wrapped with TorchRL GymWrapper.

## OpenAI Gym library
```python
from envs import GridWorldEnv
import matplotlib.pyplot as plt


env = GridWorldEnv(render_mode="rgb_array", size=10)
env.reset()
img1 = env.render()
plt.imshow(img1)
plt.title(label='GridWorldEnv: OpenAI Gym')
plt.show()
```

## Wrapped with TorchRL GymWrapper
```python
from torchrl.envs import GymWrapper


trl_env = GymWrapper(env, from_pixels=True, pixels_only=False)
td_env = trl_env.reset()
img2 = td_env['pixels']
plt.imshow(img2)
plt.title(label='GridWorldEnv: TorchRL GymWrapper')
plt.show()
```
