from __future__ import annotations

import torch
import torch.nn.functional as F

from s2p2e.trainers.common import average_logs, move_to_device


def train_actor_critic_epoch(model, loader, optimizer_actor, optimizer_critic, device, cfg):
    model.train()
    logs = []
    gamma = cfg.get("gamma", 0.99)
    torque_limit = cfg.get("torque_limit", 12.0)
    lambda_viol = cfg.get("lambda_viol", 0.5)
    for batch in loader:
        batch = move_to_device(batch, device)
        obs = batch["obs"]
        action = batch["action"]
        next_obs = batch["next_obs"]
        reward = batch["reward"]
        physics_feat = batch["physics_feat"]
        predicted_tau = batch["predicted_tau"]

        shaped_reward = model.shape_reward(reward, predicted_tau, torque_limit, lambda_viol)
        with torch.no_grad():
            next_action = model.act(next_obs, physics_feat)
            target_q = shaped_reward + gamma * model.target_critic(next_obs, next_action, physics_feat)

        current_q = model.evaluate(obs, action, physics_feat)
        critic_loss = F.mse_loss(current_q, target_q)
        optimizer_critic.zero_grad()
        critic_loss.backward()
        optimizer_critic.step()

        policy_action = model.act(obs, physics_feat)
        actor_loss = -model.evaluate(obs, policy_action, physics_feat).mean()
        feasible_feat = physics_feat
        infeasible_feat = torch.roll(physics_feat, shifts=1, dims=0)
        contrastive = model.contrastive_loss(feasible_feat, infeasible_feat)
        total_actor = actor_loss + 0.1 * contrastive
        optimizer_actor.zero_grad()
        total_actor.backward()
        optimizer_actor.step()
        model.soft_update(cfg.get("tau", 0.005))

        logs.append(
            {
                "critic_loss": float(critic_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "contrastive_loss": float(contrastive.item()),
            }
        )
    return average_logs(logs)
