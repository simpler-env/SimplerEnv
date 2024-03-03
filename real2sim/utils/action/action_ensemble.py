from collections import deque

import numpy as np


class ActionEnsembler:
    def __init__(self, pred_action_horizon, action_ensemble_temp=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_ensemble_temp = action_ensemble_temp
        self.action_history = deque(maxlen=self.pred_action_horizon)

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [pred_actions[i] for (i, pred_actions) in zip(range(num_actions - 1, -1, -1), self.action_history)]
            )
        # more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.action_ensemble_temp * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action
