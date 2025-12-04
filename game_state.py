"""
Sistema de Estado do Jogo
=========================
Gerencia o estado completo do jogo para observação por IA e serialização.
Fornece interface para treinamento de redes neurais.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import json

if TYPE_CHECKING:
    from entities import Entity


@dataclass
class ArenaConfig:
    """Configuração da arena"""
    width: int = 1200
    height: int = 800
    border: int = 50
    
    @property
    def playable_rect(self) -> Tuple[int, int, int, int]:
        """Retorna (x, y, width, height) da área jogável"""
        return (
            self.border,
            self.border,
            self.width - 2 * self.border,
            self.height - 2 * self.border
        )


@dataclass
class GameConfig:
    """Configuração do jogo para treinamento"""
    max_episode_steps: int = 3000  # ~50 segundos a 60fps
    reward_kill: float = 10.0
    reward_damage: float = 0.1
    penalty_death: float = -10.0
    penalty_damage_taken: float = -0.05
    reward_survival: float = 0.001
    
    # Para normalização
    max_speed: float = 20.0
    max_health: float = 150.0


class GameState:
    """
    Estado completo do jogo.
    Usado para observação por IA e para salvar/carregar estados.
    """
    
    def __init__(self, arena_config: ArenaConfig = None, game_config: GameConfig = None):
        self.arena = arena_config or ArenaConfig()
        self.config = game_config or GameConfig()
        
        self.entities: List['Entity'] = []
        self.step_count = 0
        self.episode_rewards: Dict[int, float] = {}  # entity_id -> total_reward
        
        # Histórico para calcular recompensas
        self._prev_states: Dict[int, Dict] = {}
        self._damage_dealt: Dict[int, float] = {}
    
    def reset(self):
        """Reseta o estado do jogo"""
        self.step_count = 0
        self.episode_rewards = {id(e): 0.0 for e in self.entities}
        self._prev_states = {id(e): e.get_state() for e in self.entities}
        self._damage_dealt = {id(e): 0.0 for e in self.entities}
    
    def add_entity(self, entity: 'Entity'):
        """Adiciona uma entidade ao estado"""
        self.entities.append(entity)
        self.episode_rewards[id(entity)] = 0.0
        self._prev_states[id(entity)] = entity.get_state()
    
    def remove_entity(self, entity: 'Entity'):
        """Remove uma entidade do estado"""
        self.entities.remove(entity)
        del self.episode_rewards[id(entity)]
        del self._prev_states[id(entity)]
    
    def step(self):
        """Avança um passo no jogo"""
        self.step_count += 1
        
        # Salvar estados atuais como anteriores para próxima iteração
        for entity in self.entities:
            self._prev_states[id(entity)] = entity.get_state()
    
    def get_observation_for_entity(self, entity: 'Entity', 
                                    enemies: List['Entity'] = None) -> Dict:
        """
        Retorna a observação do ponto de vista de uma entidade.
        """
        if enemies is None:
            enemies = [e for e in self.entities if e != entity]
        
        return {
            'self': entity.get_state(),
            'enemies': [e.get_state() for e in enemies if e.is_alive()],
            'arena': {
                'width': self.arena.width,
                'height': self.arena.height,
                'border': self.arena.border
            },
            'step': self.step_count,
            'max_steps': self.config.max_episode_steps
        }
    
    def get_observation_vector(self, entity: 'Entity', 
                               enemies: List['Entity'] = None,
                               max_enemies: int = 3) -> np.ndarray:
        """
        Retorna observação como vetor normalizado para rede neural.
        
        Estrutura do vetor:
        - [0-8]: Estado próprio (9 valores)
        - [9-16]: Inimigo 1 relativo (8 valores)
        - [17-24]: Inimigo 2 relativo (8 valores)
        - [25-32]: Inimigo 3 relativo (8 valores)
        - [33-34]: Info da arena (2 valores)
        
        Total: 35 valores
        """
        if enemies is None:
            enemies = [e for e in self.entities if e != entity and e.is_alive()]
        
        obs = []
        
        # === Estado próprio (normalizado) ===
        self_state = entity.get_state()
        arena_w, arena_h = self.arena.width, self.arena.height
        
        obs.extend([
            self_state['x'] / arena_w,
            self_state['y'] / arena_h,
            self_state['vx'] / self.config.max_speed,
            self_state['vy'] / self.config.max_speed,
            self_state['facing_angle'] / math.pi,
            self_state['health_ratio'],
            1.0 if self_state['is_invulnerable'] else 0.0,
            1.0 if self_state['weapon']['can_attack'] else 0.0 if self_state['weapon'] else 0.0,
            1.0 if self_state['ability_ready'] else 0.0
        ])
        
        # === Estados dos inimigos (relativos) ===
        # Ordenar por distância
        enemies_with_dist = []
        for e in enemies:
            if e.is_alive():
                dx = e.x - entity.x
                dy = e.y - entity.y
                dist = math.sqrt(dx*dx + dy*dy)
                enemies_with_dist.append((e, dist))
        
        enemies_with_dist.sort(key=lambda x: x[1])
        
        for i in range(max_enemies):
            if i < len(enemies_with_dist):
                enemy, dist = enemies_with_dist[i]
                enemy_state = enemy.get_state()
                
                # Posição relativa
                rel_x = (enemy_state['x'] - self_state['x']) / arena_w
                rel_y = (enemy_state['y'] - self_state['y']) / arena_h
                
                # Ângulo e distância
                angle_to = math.atan2(rel_y * arena_h, rel_x * arena_w)
                norm_dist = dist / math.sqrt(arena_w**2 + arena_h**2)
                
                obs.extend([
                    rel_x,
                    rel_y,
                    norm_dist,
                    angle_to / math.pi,
                    enemy_state['vx'] / self.config.max_speed,
                    enemy_state['vy'] / self.config.max_speed,
                    enemy_state['health_ratio'],
                    1.0 if enemy_state['weapon']['is_attacking'] else 0.0 if enemy_state['weapon'] else 0.0
                ])
            else:
                # Padding para inimigos ausentes
                obs.extend([0.0] * 8)
        
        # === Info da arena ===
        # Distância normalizada das bordas
        border_dist_x = min(entity.x - self.arena.border, 
                          self.arena.width - self.arena.border - entity.x) / (arena_w / 2)
        border_dist_y = min(entity.y - self.arena.border,
                          self.arena.height - self.arena.border - entity.y) / (arena_h / 2)
        
        obs.extend([
            border_dist_x,
            border_dist_y
        ])
        
        return np.array(obs, dtype=np.float32)
    
    @staticmethod
    def get_observation_size(max_enemies: int = 3) -> int:
        """Retorna o tamanho do vetor de observação"""
        return 9 + (8 * max_enemies) + 2
    
    @staticmethod
    def get_action_size() -> int:
        """Retorna o tamanho do vetor de ações"""
        return 4  # move_x, move_y, attack, ability
    
    def calculate_reward(self, entity: 'Entity', 
                        enemies: List['Entity'] = None) -> float:
        """
        Calcula a recompensa para uma entidade neste passo.
        """
        entity_id = id(entity)
        prev_state = self._prev_states.get(entity_id, {})
        curr_state = entity.get_state()
        
        if enemies is None:
            enemies = [e for e in self.entities if e != entity]
        
        reward = 0.0
        
        # Recompensa por dano causado
        for enemy in enemies:
            if hasattr(enemy, 'last_damage_source') and enemy.last_damage_source == entity:
                # Dano foi causado neste frame
                damage = prev_state.get('health', 0) - enemy.health if hasattr(enemy, 'health') else 0
                reward += self.config.reward_damage * abs(damage)
                enemy.last_damage_source = None  # Reset
        
        # Penalidade por dano recebido
        if prev_state:
            health_diff = curr_state['health'] - prev_state.get('health', curr_state['health'])
            if health_diff < 0:
                reward += self.config.penalty_damage_taken * abs(health_diff)
        
        # Recompensa por matar inimigo
        for enemy in enemies:
            if not enemy.is_alive():
                if hasattr(enemy, '_was_alive') and enemy._was_alive:
                    reward += self.config.reward_kill
                    enemy._was_alive = False
        
        # Penalidade por morrer
        if not entity.is_alive():
            if prev_state.get('is_alive', True):
                reward += self.config.penalty_death
        
        # Pequena recompensa por sobreviver
        if entity.is_alive():
            reward += self.config.reward_survival
        
        # Acumular recompensa total
        self.episode_rewards[entity_id] = self.episode_rewards.get(entity_id, 0) + reward
        
        return reward
    
    def is_episode_done(self) -> bool:
        """Verifica se o episódio terminou"""
        # Tempo máximo atingido
        if self.step_count >= self.config.max_episode_steps:
            return True
        
        # Apenas um ou nenhum sobrevivente
        alive_count = sum(1 for e in self.entities if e.is_alive())
        if alive_count <= 1:
            return True
        
        return False
    
    def get_winner(self) -> Optional['Entity']:
        """Retorna o vencedor (se houver)"""
        alive = [e for e in self.entities if e.is_alive()]
        if len(alive) == 1:
            return alive[0]
        return None
    
    def to_dict(self) -> Dict:
        """Serializa o estado para dict"""
        return {
            'arena': {
                'width': self.arena.width,
                'height': self.arena.height,
                'border': self.arena.border
            },
            'entities': [e.get_state() for e in self.entities],
            'step_count': self.step_count,
            'episode_rewards': {str(k): v for k, v in self.episode_rewards.items()}
        }
    
    def to_json(self) -> str:
        """Serializa o estado para JSON"""
        return json.dumps(self.to_dict(), indent=2)


class TrainingEnvironment:
    """
    Ambiente de treinamento compatível com OpenAI Gym.
    Facilita integração com bibliotecas de RL como Stable-Baselines3.
    """
    
    def __init__(self, arena_config: ArenaConfig = None, 
                 game_config: GameConfig = None):
        self.arena = arena_config or ArenaConfig()
        self.config = game_config or GameConfig()
        self.game_state = GameState(self.arena, self.config)
        
        # Entidades do ambiente
        self.agent: Optional['Entity'] = None
        self.opponents: List['Entity'] = []
        
        # Espaços de observação e ação
        self.observation_size = GameState.get_observation_size()
        self.action_size = GameState.get_action_size()
    
    def reset(self, agent_class: str = "warrior", 
              opponent_classes: List[str] = None) -> np.ndarray:
        """
        Reseta o ambiente para um novo episódio.
        
        Args:
            agent_class: Classe do agente sendo treinado
            opponent_classes: Classes dos oponentes
        
        Returns:
            Observação inicial
        """
        from entities import ClassRegistry
        import random
        
        self.game_state = GameState(self.arena, self.config)
        
        # Criar agente
        spawn_x = random.randint(100, self.arena.width - 100)
        spawn_y = random.randint(100, self.arena.height - 100)
        self.agent = ClassRegistry.create(agent_class, spawn_x, spawn_y, (100, 200, 255))
        self.agent._was_alive = True
        self.game_state.add_entity(self.agent)
        
        # Criar oponentes
        self.opponents = []
        opponent_classes = opponent_classes or ["warrior"]
        
        for i, opp_class in enumerate(opponent_classes):
            # Spawn em posição diferente
            opp_x = random.randint(100, self.arena.width - 100)
            opp_y = random.randint(100, self.arena.height - 100)
            
            # Garantir distância mínima
            while math.sqrt((opp_x - spawn_x)**2 + (opp_y - spawn_y)**2) < 200:
                opp_x = random.randint(100, self.arena.width - 100)
                opp_y = random.randint(100, self.arena.height - 100)
            
            opponent = ClassRegistry.create(opp_class, opp_x, opp_y, (255, 100, 100))
            opponent._was_alive = True
            self.opponents.append(opponent)
            self.game_state.add_entity(opponent)
        
        self.game_state.reset()
        
        return self.game_state.get_observation_vector(self.agent, self.opponents)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Executa um passo no ambiente.
        
        Args:
            action: Vetor de ações [move_x, move_y, attack, ability]
        
        Returns:
            observation: Nova observação
            reward: Recompensa
            done: Se o episódio terminou
            info: Informações adicionais
        """
        # Aplicar ação do agente
        move_x = float(np.clip(action[0], -1, 1))
        move_y = float(np.clip(action[1], -1, 1))
        attack = bool(action[2] > 0.5)
        ability = bool(action[3] > 0.5)
        
        if move_x != 0 or move_y != 0:
            self.agent.move(move_x, move_y)
        
        if attack:
            self.agent.attack()
        
        if ability:
            self.agent.use_ability()
        
        # Atualizar oponentes (precisam de controladores)
        for opponent in self.opponents:
            if opponent.controller:
                opponent.controller.update(1/60)  # dt fixo
        
        # Avançar estado
        self.game_state.step()
        
        # Calcular recompensa
        reward = self.game_state.calculate_reward(self.agent, self.opponents)
        
        # Verificar se terminou
        done = self.game_state.is_episode_done()
        
        # Info adicional
        info = {
            'step': self.game_state.step_count,
            'agent_health': self.agent.health,
            'opponents_alive': sum(1 for o in self.opponents if o.is_alive()),
            'winner': self.game_state.get_winner()
        }
        
        # Nova observação
        obs = self.game_state.get_observation_vector(self.agent, self.opponents)
        
        return obs, reward, done, info
    
    def render(self, screen=None):
        """Renderiza o ambiente (opcional)"""
        if screen is None:
            return
        
        # Desenhar arena
        import pygame
        screen.fill((30, 30, 30))
        
        arena_rect = pygame.Rect(*self.arena.playable_rect)
        pygame.draw.rect(screen, (50, 50, 50), arena_rect)
        pygame.draw.rect(screen, (255, 255, 255), arena_rect, 2)
        
        # Desenhar entidades
        for entity in self.game_state.entities:
            entity.draw(screen)
    
    def close(self):
        """Fecha o ambiente"""
        pass
