"""
Sistema de Treinamento de Redes Neurais
=======================================
Treina agentes de IA para jogar Circle Warriors usando Reinforcement Learning.

Requisitos:
    pip install torch numpy gymnasium stable-baselines3

Uso básico:
    python train.py                     # Treina com configurações padrão
    python train.py --episodes 10000    # Treina por 10000 episódios
    python train.py --render            # Treina visualizando o jogo
    python train.py --test model.pth    # Testa um modelo treinado
"""

import os
import sys
import math
import random
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import json
from datetime import datetime

# Adicionar o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports do jogo
from entities import ClassRegistry, Entity
from weapons import WeaponRegistry
from physics import Physics
from controller import AIController, SimpleAI, StrategicAI
from game_state import GameState, ArenaConfig, GameConfig


# ============================================================================
# AMBIENTE DE TREINAMENTO (Compatível com Gymnasium/Gym)
# ============================================================================

class CircleWarriorsEnv:
    """
    Ambiente de treinamento para Circle Warriors.
    Compatível com a interface do Gymnasium (antigo OpenAI Gym).
    """
    
    def __init__(self, 
                 agent_class: str = "warrior",
                 agent_weapon: str = "sword",
                 opponent_class: str = "warrior",
                 opponent_weapon: str = "sword",
                 opponent_ai: str = "simple",
                 render_mode: str = None,
                 max_steps: int = 3000):
        """
        Args:
            agent_class: Classe do agente sendo treinado
            agent_weapon: Arma do agente
            opponent_class: Classe do oponente
            opponent_weapon: Arma do oponente
            opponent_ai: Tipo de IA do oponente ("simple", "random", "none")
            render_mode: "human" para visualizar, None para treinar rápido
            max_steps: Máximo de passos por episódio
        """
        self.agent_class = agent_class
        self.agent_weapon = agent_weapon
        self.opponent_class = opponent_class
        self.opponent_weapon = opponent_weapon
        self.opponent_ai_type = opponent_ai
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Configuração da arena
        self.arena_config = ArenaConfig(800, 600, 50)
        self.game_config = GameConfig(max_episode_steps=max_steps)
        
        # Física
        self.physics = Physics(self.arena_config.width, self.arena_config.height)
        
        # Entidades
        self.agent: Optional[Entity] = None
        self.opponent: Optional[Entity] = None
        self.opponent_controller: Optional[AIController] = None
        
        # Estado
        self.steps = 0
        self.game_state: Optional[GameState] = None
        
        # Pygame (só inicializa se render_mode == "human")
        self.screen = None
        self.clock = None
        if render_mode == "human":
            self._init_pygame()
        
        # Espaços de observação e ação
        self.observation_size = self._get_observation_size()
        self.action_size = 4  # move_x, move_y, attack, ability
        
        # Histórico para métricas
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _init_pygame(self):
        """Inicializa pygame para renderização"""
        import pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.arena_config.width, self.arena_config.height)
        )
        pygame.display.set_caption("Circle Warriors - Training")
        self.clock = pygame.time.Clock()
    
    def _get_observation_size(self) -> int:
        """Retorna o tamanho do vetor de observação"""
        # Estado próprio: 9 valores
        # Estado do oponente: 8 valores
        # Distância das bordas: 2 valores
        return 19
    
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Reseta o ambiente para um novo episódio.
        
        Returns:
            observation: Observação inicial
            info: Informações adicionais
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.steps = 0
        self.game_state = GameState(self.arena_config, self.game_config)
        
        # Posições aleatórias
        margin = 100
        agent_x = random.randint(margin, self.arena_config.width // 2 - 50)
        agent_y = random.randint(margin, self.arena_config.height - margin)
        
        opponent_x = random.randint(self.arena_config.width // 2 + 50, 
                                    self.arena_config.width - margin)
        opponent_y = random.randint(margin, self.arena_config.height - margin)
        
        # Criar agente
        self.agent = ClassRegistry.create(
            self.agent_class, agent_x, agent_y, (100, 200, 100)
        )
        self.agent.set_weapon(self.agent_weapon)
        self.agent._was_alive = True
        self.agent._prev_health = self.agent.health
        self.game_state.add_entity(self.agent)
        
        # Criar oponente
        self.opponent = ClassRegistry.create(
            self.opponent_class, opponent_x, opponent_y, (255, 100, 100)
        )
        self.opponent.set_weapon(self.opponent_weapon)
        self.opponent._was_alive = True
        self.opponent._prev_health = self.opponent.health
        self.game_state.add_entity(self.opponent)
        
        # Configurar IA do oponente
        if self.opponent_ai_type == "simple":
            # Usa StrategicAI com estratégia específica para a combinação classe+arma
            self.opponent_controller = StrategicAI(
                class_name=self.opponent_class.capitalize(),
                weapon_name=self.opponent_weapon.capitalize()
            )
            self.opponent_controller.entity = self.opponent
            self.opponent_controller.set_targets([self.agent])
        elif self.opponent_ai_type == "random":
            self.opponent_controller = RandomAI()
            self.opponent_controller.entity = self.opponent
        else:
            self.opponent_controller = None
        
        self.game_state.reset()
        
        obs = self._get_observation()
        info = {"agent_health": self.agent.health, "opponent_health": self.opponent.health}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Executa um passo no ambiente.
        
        Args:
            action: [move_x, move_y, attack, ability] valores entre -1 e 1
        
        Returns:
            observation: Nova observação
            reward: Recompensa
            terminated: Se o episódio terminou (vitória/derrota)
            truncated: Se o episódio foi cortado (tempo)
            info: Informações adicionais
        """
        self.steps += 1
        dt = 1/60  # 60 FPS
        
        # Salvar estados anteriores
        prev_agent_health = self.agent.health
        prev_opponent_health = self.opponent.health
        
        # Aplicar ação do agente
        self._apply_action(self.agent, action)
        
        # Atualizar oponente
        if self.opponent_controller and self.opponent.is_alive():
            self.opponent_controller.update(dt)
        
        # Atualizar entidades
        for entity in [self.agent, self.opponent]:
            if entity.is_alive():
                entity.update(dt)
        
        # Física
        self.physics.handle_collisions([self.agent, self.opponent])
        
        # Limitar à arena
        import pygame
        arena_rect = pygame.Rect(*self.arena_config.playable_rect)
        for entity in [self.agent, self.opponent]:
            self.physics.constrain_to_arena(entity, arena_rect)
        
        # Calcular recompensa
        reward = self._calculate_reward(
            prev_agent_health, prev_opponent_health
        )
        
        # Verificar término
        terminated = False
        truncated = False
        
        if not self.agent.is_alive():
            terminated = True
            reward -= 10  # Penalidade por morrer
        elif not self.opponent.is_alive():
            terminated = True
            reward += 10  # Bônus por vencer
        elif self.steps >= self.max_steps:
            truncated = True
        
        # Renderizar se necessário
        if self.render_mode == "human":
            self.render()
        
        obs = self._get_observation()
        info = {
            "agent_health": self.agent.health,
            "opponent_health": self.opponent.health,
            "steps": self.steps,
            "damage_dealt": prev_opponent_health - self.opponent.health,
            "damage_taken": prev_agent_health - self.agent.health
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, entity: Entity, action: np.ndarray):
        """Aplica uma ação a uma entidade"""
        # Movimento
        move_x = float(np.clip(action[0], -1, 1))
        move_y = float(np.clip(action[1], -1, 1))
        
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            entity.move(move_x, move_y)
        else:
            entity.moving = False
        
        # Ataque (threshold de 0.5)
        if action[2] > 0.5:
            entity.attack()
        
        # Habilidade
        if action[3] > 0.5:
            entity.use_ability()
    
    def _get_observation(self) -> np.ndarray:
        """Retorna a observação atual normalizada"""
        obs = []
        
        # Estado do agente
        obs.extend([
            self.agent.x / self.arena_config.width,
            self.agent.y / self.arena_config.height,
            self.agent.vx / 20,
            self.agent.vy / 20,
            self.agent.facing_angle / math.pi,
            self.agent.health / self.agent.stats_manager.get_stats().max_health,
            1.0 if self.agent.invulnerable_time > 0 else 0.0,
            1.0 if self.agent.weapon and self.agent.weapon.can_attack else 0.0,
            1.0 if self.agent.ability_cooldown <= 0 else 0.0
        ])
        
        # Estado do oponente (relativo)
        rel_x = (self.opponent.x - self.agent.x) / self.arena_config.width
        rel_y = (self.opponent.y - self.agent.y) / self.arena_config.height
        distance = math.sqrt(rel_x**2 + rel_y**2)
        angle_to = math.atan2(rel_y, rel_x)
        
        obs.extend([
            rel_x,
            rel_y,
            distance,
            angle_to / math.pi,
            self.opponent.vx / 20,
            self.opponent.vy / 20,
            self.opponent.health / self.opponent.stats_manager.get_stats().max_health,
            1.0 if self.opponent.weapon and self.opponent.weapon.is_attacking else 0.0
        ])
        
        # Distância das bordas
        border_x = min(
            self.agent.x - self.arena_config.border,
            self.arena_config.width - self.arena_config.border - self.agent.x
        ) / (self.arena_config.width / 2)
        border_y = min(
            self.agent.y - self.arena_config.border,
            self.arena_config.height - self.arena_config.border - self.agent.y
        ) / (self.arena_config.height / 2)
        
        obs.extend([border_x, border_y])
        
        return np.array(obs, dtype=np.float32)
    
    def _calculate_reward(self, prev_agent_health: float, 
                          prev_opponent_health: float) -> float:
        """Calcula a recompensa do passo atual"""
        reward = 0.0
        
        # Recompensa por dano causado
        damage_dealt = prev_opponent_health - self.opponent.health
        if damage_dealt > 0:
            reward += damage_dealt * 0.1
        
        # Penalidade por dano recebido
        damage_taken = prev_agent_health - self.agent.health
        if damage_taken > 0:
            reward -= damage_taken * 0.05
        
        # Pequena recompensa por sobreviver
        reward += 0.001
        
        # Recompensa por estar perto do oponente (encoraja combate)
        distance = math.sqrt(
            (self.agent.x - self.opponent.x)**2 + 
            (self.agent.y - self.opponent.y)**2
        )
        if distance < 150:
            reward += 0.002
        
        return reward
    
    def render(self):
        """Renderiza o jogo"""
        if self.screen is None:
            return
        
        import pygame
        
        # Processar eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
        
        # Fundo
        self.screen.fill((30, 30, 30))
        
        # Arena
        arena_rect = pygame.Rect(*self.arena_config.playable_rect)
        pygame.draw.rect(self.screen, (50, 50, 50), arena_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), arena_rect, 2)
        
        # Entidades
        for entity in [self.agent, self.opponent]:
            if entity:
                entity.draw(self.screen)
        
        # Info
        font = pygame.font.Font(None, 24)
        info_text = f"Step: {self.steps} | Agent HP: {int(self.agent.health)} | Enemy HP: {int(self.opponent.health)}"
        text_surf = font.render(info_text, True, (255, 255, 255))
        self.screen.blit(text_surf, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """Fecha o ambiente"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None


class RandomAI(AIController):
    """IA que toma ações aleatórias (para treino inicial)"""
    
    def decide_actions(self, observation: Dict) -> Dict:
        return {
            'move_x': random.uniform(-1, 1),
            'move_y': random.uniform(-1, 1),
            'attack': random.random() > 0.7,
            'ability': random.random() > 0.95
        }


# ============================================================================
# REDE NEURAL (PyTorch)
# ============================================================================

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch não encontrado. Instale com: pip install torch")


if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Rede neural Actor-Critic para PPO"""
        
        def __init__(self, obs_size: int, action_size: int, hidden_size: int = 256):
            super().__init__()
            
            # Camadas compartilhadas
            self.shared = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            
            # Actor (política) - saída contínua para movimento, discreta para ações
            self.actor_mean = nn.Linear(hidden_size, action_size)
            self.actor_log_std = nn.Parameter(torch.zeros(action_size))
            
            # Critic (valor)
            self.critic = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            shared = self.shared(x)
            
            # Actor
            action_mean = torch.tanh(self.actor_mean(shared))
            action_std = torch.exp(self.actor_log_std)
            
            # Critic
            value = self.critic(shared)
            
            return action_mean, action_std, value
        
        def get_action(self, obs: np.ndarray, deterministic: bool = False):
            """Obtém uma ação dado uma observação"""
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_mean, action_std, value = self.forward(obs_tensor)
                
                if deterministic:
                    action = action_mean
                else:
                    # Amostra de distribuição normal
                    dist = torch.distributions.Normal(action_mean, action_std)
                    action = dist.sample()
                
                return action.squeeze().numpy(), value.item()
        
        def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
            """Avalia ações para treinamento"""
            action_mean, action_std, values = self.forward(obs)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            
            return values.squeeze(), log_probs, entropy


    class PPOTrainer:
        """Treinador usando Proximal Policy Optimization"""
        
        def __init__(self, env: CircleWarriorsEnv, 
                     hidden_size: int = 256,
                     lr: float = 3e-4,
                     gamma: float = 0.99,
                     gae_lambda: float = 0.95,
                     clip_epsilon: float = 0.2,
                     epochs_per_update: int = 10,
                     batch_size: int = 64):
            
            self.env = env
            self.gamma = gamma
            self.gae_lambda = gae_lambda
            self.clip_epsilon = clip_epsilon
            self.epochs_per_update = epochs_per_update
            self.batch_size = batch_size
            
            # Rede neural
            self.model = ActorCritic(
                env.observation_size, 
                env.action_size, 
                hidden_size
            )
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            
            # Buffer de experiência
            self.obs_buffer = []
            self.action_buffer = []
            self.reward_buffer = []
            self.value_buffer = []
            self.log_prob_buffer = []
            self.done_buffer = []
            
            # Métricas
            self.episode_rewards = []
            self.episode_lengths = []
        
        def collect_rollout(self, num_steps: int = 2048):
            """Coleta experiências do ambiente"""
            self.obs_buffer.clear()
            self.action_buffer.clear()
            self.reward_buffer.clear()
            self.value_buffer.clear()
            self.log_prob_buffer.clear()
            self.done_buffer.clear()
            
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            for _ in range(num_steps):
                # Obter ação
                action, value = self.model.get_action(obs)
                
                # Calcular log prob
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action_tensor = torch.FloatTensor(action).unsqueeze(0)
                    _, log_prob, _ = self.model.evaluate_actions(obs_tensor, action_tensor)
                
                # Executar ação
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Armazenar
                self.obs_buffer.append(obs)
                self.action_buffer.append(action)
                self.reward_buffer.append(reward)
                self.value_buffer.append(value)
                self.log_prob_buffer.append(log_prob.item())
                self.done_buffer.append(done)
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    episode_reward = 0
                    episode_length = 0
                    obs, _ = self.env.reset()
                else:
                    obs = next_obs
            
            return self._compute_advantages()
        
        def _compute_advantages(self):
            """Calcula vantagens usando GAE"""
            advantages = []
            returns = []
            gae = 0
            
            # Calcular de trás para frente
            for t in reversed(range(len(self.reward_buffer))):
                if t == len(self.reward_buffer) - 1:
                    next_value = 0
                else:
                    next_value = self.value_buffer[t + 1]
                
                if self.done_buffer[t]:
                    next_value = 0
                    gae = 0
                
                delta = self.reward_buffer[t] + self.gamma * next_value - self.value_buffer[t]
                gae = delta + self.gamma * self.gae_lambda * gae
                
                advantages.insert(0, gae)
                returns.insert(0, gae + self.value_buffer[t])
            
            return np.array(advantages), np.array(returns)
        
        def update(self, advantages: np.ndarray, returns: np.ndarray):
            """Atualiza a rede neural"""
            # Converter para tensores
            obs_tensor = torch.FloatTensor(np.array(self.obs_buffer))
            action_tensor = torch.FloatTensor(np.array(self.action_buffer))
            old_log_probs = torch.FloatTensor(np.array(self.log_prob_buffer))
            advantages_tensor = torch.FloatTensor(advantages)
            returns_tensor = torch.FloatTensor(returns)
            
            # Normalizar vantagens
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            
            # Múltiplas épocas de atualização
            dataset_size = len(self.obs_buffer)
            
            for _ in range(self.epochs_per_update):
                # Embaralhar índices
                indices = np.random.permutation(dataset_size)
                
                for start in range(0, dataset_size, self.batch_size):
                    end = start + self.batch_size
                    batch_indices = indices[start:end]
                    
                    # Mini-batch
                    batch_obs = obs_tensor[batch_indices]
                    batch_actions = action_tensor[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_advantages = advantages_tensor[batch_indices]
                    batch_returns = returns_tensor[batch_indices]
                    
                    # Forward pass
                    values, log_probs, entropy = self.model.evaluate_actions(
                        batch_obs, batch_actions
                    )
                    
                    # Ratio para PPO
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    
                    # Clipped surrogate loss
                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = F.mse_loss(values, batch_returns)
                    
                    # Entropy bonus
                    entropy_loss = -entropy.mean()
                    
                    # Loss total
                    loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                    
                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
        
        def train(self, total_timesteps: int = 100000, 
                  save_freq: int = 10000,
                  log_freq: int = 1000,
                  save_path: str = "models"):
            """Loop principal de treinamento"""
            os.makedirs(save_path, exist_ok=True)
            
            timesteps = 0
            iteration = 0
            
            print(f"\n{'='*60}")
            print(f"Iniciando treinamento PPO")
            print(f"Total timesteps: {total_timesteps}")
            print(f"{'='*60}\n")
            
            while timesteps < total_timesteps:
                iteration += 1
                
                # Coletar experiências
                advantages, returns = self.collect_rollout(num_steps=2048)
                timesteps += len(self.obs_buffer)
                
                # Atualizar rede
                self.update(advantages, returns)
                
                # Log
                if len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-100:]
                    recent_lengths = self.episode_lengths[-100:]
                    
                    print(f"Iteration {iteration} | "
                          f"Timesteps: {timesteps} | "
                          f"Mean Reward: {np.mean(recent_rewards):.2f} | "
                          f"Mean Length: {np.mean(recent_lengths):.0f}")
                
                # Salvar modelo
                if timesteps % save_freq < 2048:
                    self.save(f"{save_path}/model_{timesteps}.pth")
            
            # Salvar modelo final
            self.save(f"{save_path}/model_final.pth")
            print(f"\nTreinamento concluído! Modelo salvo em {save_path}/model_final.pth")
        
        def save(self, path: str):
            """Salva o modelo"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'episode_rewards': self.episode_rewards,
                'episode_lengths': self.episode_lengths
            }, path)
        
        def load(self, path: str):
            """Carrega um modelo"""
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.episode_lengths = checkpoint.get('episode_lengths', [])


# ============================================================================
# FUNÇÕES DE UTILIDADE
# ============================================================================

def test_model(model_path: str, 
               agent_class: str = "warrior",
               agent_weapon: str = "sword",
               num_episodes: int = 10):
    """Testa um modelo treinado"""
    if not TORCH_AVAILABLE:
        print("PyTorch necessário para testar modelos!")
        return
    
    env = CircleWarriorsEnv(
        agent_class=agent_class,
        agent_weapon=agent_weapon,
        opponent_ai="simple",
        render_mode="human"
    )
    
    # Carregar modelo
    model = ActorCritic(env.observation_size, env.action_size)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    wins = 0
    total_reward = 0
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.get_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        total_reward += episode_reward
        
        if info['opponent_health'] <= 0:
            wins += 1
            print(f"Episode {episode + 1}: VITÓRIA! Reward: {episode_reward:.2f}")
        else:
            print(f"Episode {episode + 1}: Derrota. Reward: {episode_reward:.2f}")
    
    print(f"\nResultados: {wins}/{num_episodes} vitórias ({100*wins/num_episodes:.1f}%)")
    print(f"Reward médio: {total_reward/num_episodes:.2f}")
    
    env.close()


def train_with_stable_baselines():
    """
    Exemplo de como usar Stable-Baselines3 para treinar.
    Requer: pip install stable-baselines3
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv
        import gymnasium as gym
        
        # Wrapper para compatibilidade com Gymnasium
        class GymWrapper(gym.Env):
            def __init__(self):
                super().__init__()
                self.env = CircleWarriorsEnv(render_mode=None)
                
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.env.observation_size,),
                    dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1, high=1,
                    shape=(self.env.action_size,),
                    dtype=np.float32
                )
            
            def reset(self, seed=None, options=None):
                return self.env.reset(seed)
            
            def step(self, action):
                return self.env.step(action)
            
            def render(self):
                return self.env.render()
            
            def close(self):
                return self.env.close()
        
        print("Treinando com Stable-Baselines3...")
        
        # Criar ambiente vetorizado
        env = DummyVecEnv([lambda: GymWrapper()])
        
        # Criar modelo PPO
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./logs/"
        )
        
        # Treinar
        model.learn(total_timesteps=100000)
        
        # Salvar
        model.save("models/sb3_model")
        print("Modelo salvo em models/sb3_model")
        
    except ImportError:
        print("Stable-Baselines3 não encontrado!")
        print("Instale com: pip install stable-baselines3 gymnasium")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Treinar IA para Circle Warriors")
    
    parser.add_argument("--episodes", type=int, default=100000,
                        help="Número de timesteps para treinar")
    parser.add_argument("--render", action="store_true",
                        help="Visualizar o treinamento")
    parser.add_argument("--test", type=str, default=None,
                        help="Caminho do modelo para testar")
    parser.add_argument("--class", dest="agent_class", type=str, default="warrior",
                        help="Classe do agente")
    parser.add_argument("--weapon", type=str, default="sword",
                        help="Arma do agente")
    parser.add_argument("--sb3", action="store_true",
                        help="Usar Stable-Baselines3 ao invés de PPO próprio")
    
    args = parser.parse_args()
    
    if args.test:
        # Modo de teste
        test_model(args.test, args.agent_class, args.weapon)
    elif args.sb3:
        # Treinar com Stable-Baselines3
        train_with_stable_baselines()
    else:
        # Treinar com implementação própria
        if not TORCH_AVAILABLE:
            print("PyTorch é necessário para treinar!")
            print("Instale com: pip install torch")
            return
        
        env = CircleWarriorsEnv(
            agent_class=args.agent_class,
            agent_weapon=args.weapon,
            opponent_ai="simple",
            render_mode="human" if args.render else None
        )
        
        trainer = PPOTrainer(env)
        trainer.train(total_timesteps=args.episodes)
        
        env.close()


if __name__ == "__main__":
    main()
