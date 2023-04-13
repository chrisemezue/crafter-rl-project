# crafter attention maps

Based on this [task](https://www.notion.so/Work-on-metric-to-quantitatively-evaluate-attention-maps-9e85515a233f4426a92fb84e52fe4b26), I am saving the attention maps and episode details at each evaluation step.

1. The parent folder is called `crafter-attn-maps`
2. Inside the folder is a sub folder, which is the wandb tag name.
3. Inside is yet again another folder, named by the actual wandb name of the experiment. This is done to prevent overwriting if there are multiple experiments done with the same wandb tag name.
4. Then inside are two folders `valid_det` corresponding to evaluation on a deterministic env and `valid_sto` for the stochastic environment.
5. Inside each of these folders are the actual files. The files are **pickled** so you need to unpickle them to retrieve their actual data. Read below to learn what each file represents.

#### File details
The numbers at the end of each file represents the timestep of the evaluation.

`attn_maps` => refers to the attention maps fresh out of the object centric model.
`slot_masks` => refers to the same attention maps, but they have been made to fit the dimension of the environment. The shape of this usually fits the shape of the episode observation environment (`episode_observations`).
`episode_details` => is a dict containing the list of episode rewards and episode lengths. I figured it would help to have these saved to foster our analysis. 
> If there are more details that need to be saved for the later analysis, I plan to save them in this dict.

`episode_details` is currently of the form
```python
{
'episode_rewards': List
'episode_lengths': List
}
```