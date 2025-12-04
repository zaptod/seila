# test_team.py - Script de teste para o sistema de equipes
import numpy as np
from team_train import TeamBattleEnv, TEAM_COMPOSITIONS

print("Iniciando teste do TeamBattleEnv...")

# Teste rápido sem renderização
env = TeamBattleEnv(
    team_size=2,
    blue_team_config=TEAM_COMPOSITIONS['2v2_standard'],
    red_team_config=TEAM_COMPOSITIONS['2v2_aggressive'],
    render_mode=None,
    max_steps=500
)

obs, info = env.reset()
print(f"Observacao shape: {obs[0].shape}")
print(f"Blue alive: {info['blue_alive']}, Red alive: {info['red_alive']}")

# Simular alguns passos
for step in range(100):
    actions = [np.random.uniform(-1, 1, 4) for _ in range(2)]
    obs, rewards, terminated, truncated, info = env.step(actions)
    if terminated or truncated:
        print(f"Episodio terminou no step {step}")
        break

print(f"Apos simulacao: Blue {info['blue_alive']} vs Red {info['red_alive']}")
print(f"Steps: {info['steps']}")
print("Teste concluido com sucesso!")
