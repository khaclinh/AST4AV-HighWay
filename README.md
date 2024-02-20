# A novel framework for adaptive stress testing of autonomous vehicles in the highway

---

## Project Structure
- [`config`](./config) Dir containing config files for training/evaluating.
- [`highway_env`](./highway_env) Dir containing source code.
- [`weights`](./weights): Dir storing trained weights.
- [`LICENSE`](./LICENSE): File describing license terms.
- [`main.py`](./main.py): Main file.
- [`README.md`](./README.md): This file!


---

## Installation
Please following instruction from [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) for installation.


### Config
[config](./config) consist of:
* `env_name`: class name of simulation environment.
* `num_episodes`: number of episode to run reinforcement learning for AST.
* `num_steps`: number of each step in each episode.
* `save_result_csv`: path to a csv file to store the recorded collision. The format of each collision is described as below:
| `#episode` | `#step` | `ego_speed` | `ego_acceleration` | `ego_ast_action` | `lane_index` | `speed_crashed_veh` | `acceleration_crashed_veh` | `crashed_lane_index` | `crashed_ast_action` | `crashed_front` | `crashed_distance` |
* `model_path`: path to model file to be saved after training reinforcement learning model for AST.
* `reward_file` (Optional): path to save training reward for further analysis.
* `save_pic`: path to folder for exporting scene from simulation for further analysis.
* `ttc_threshold`: a threshold of time-to-collision that used to calculate propability of collision as in paper.
* `ego_collision_weight`: the value of `lambda`.

### Train and testing
To train and test a model, simply run
```shell
python main.py --config_path config/<YOUR_CONFIG>.yaml
```

- `config_path`: Path to trainer config file, containing details for experiment as described above...

---
  
## Citation
If you used the code in this repository or found the paper interesting, please cite it as
```text
@misc{trinh2024,
      title={A novel framework for adaptive stress testing of autonomous vehicles in highways}, 
      author={Linh Trinh and Quang-Hung Luu and Thai M. Nguyen and Hai L. Vu},
      year={2024},
      eprint={2402.11813},
      archivePrefix={arXiv}
}
```

---

## Acknowledgements
- Especially thanks to [HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv) with there excellent work. This code base borrow borrow frameworks to accelerate implementation.
- This project is supported by a grant from the Smart Pavements Australia Research Collaboration Hub.

## Contact
If you have any problems about this work, please contact Linh Trinh at linhtk.dhbk@gmail.com.

## Licence
This project is licenced under the `Commons Clause` and `GNU GPL` licenses.
For commercial use, please contact the authors. 

---
