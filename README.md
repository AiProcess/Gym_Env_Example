# Grid World Environment
An Example for creating environment using OpenAI Gym. Then this environment is wrapped with TorchRL GymWrapper.

```python
env = GridWorldEnv(render_mode="rgb_array", size=10)
env.reset()
img1 = env.render()
plt.imshow(img1)
plt.title(label='GridWorldEnv: OpenAI Gym')
plt.show()
```
