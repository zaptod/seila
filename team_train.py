"""
Sistema de Treinamento em Grupo - Torneio de Redes Neurais
===========================================================
Treina equipes de agentes para combate cooperativo.

Modos de Treinamento:
- HEALER: Treina para manter aliados vivos
- TANK: Treina para absorver dano e proteger
- DPS: Treina para causar dano máximo
- CONTROLLER: Treina para aplicar controle de grupo
- SUPPORT: Treina para buffar e auxiliar aliados

Formatos de Jogo:
- 2v2: Combate em duplas
- 3v3: Combate em trios
- 5v5: Combate em equipes completas

Requisitos:
    pip install torch numpy pygame
"""

import os
import sys
import math
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import json
from datetime import datetime

# Adicionar diretório ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports do jogo
from entities import ClassRegistry, Entity
from weapons import WeaponRegistry
from physics import Physics
from controller import AIController, StrategicAI, CLASS_WEAPON_STRATEGIES
from game_state import GameState, ArenaConfig, GameConfig
from stats import StatusEffectType


# ============================================================================
# ENUMS E CONFIGURAÇÕES
# ============================================================================

class TeamRole(Enum):
    """Papéis dentro de uma equipe"""
    HEALER = "healer"           # Cleric, Enchanter com Staff/Tome
    TANK = "tank"               # Guardian, Tank com Shield
    DPS_MELEE = "dps_melee"     # Warrior, Berserker, Assassin
    DPS_RANGED = "dps_ranged"   # Ranger com Bow
    CONTROLLER = "controller"   # Controller, Trapper
    SUPPORT = "support"         # Enchanter, classes com buffs


class TrainingMode(Enum):
    """Modos de treinamento especializados"""
    STANDARD = "standard"           # Treinamento padrão (maximize vitória)
    HEALER_FOCUSED = "healer"       # Recompensa por cura e manter aliados
    TANK_FOCUSED = "tank"           # Recompensa por absorver dano/proteger
    DPS_FOCUSED = "dps"             # Recompensa por dano causado
    CC_FOCUSED = "controller"       # Recompensa por tempo de CC
    SUPPORT_FOCUSED = "support"     # Recompensa por buffs aplicados
    SURVIVAL = "survival"           # Sobreviver o máximo possível
    OBJECTIVE = "objective"         # Controle de objetivo (futuro)


@dataclass
class TeamConfig:
    """Configuração de uma equipe"""
    name: str
    color: Tuple[int, int, int]
    members: List[Dict]  # Lista de {class, weapon, role}
    spawn_side: str = "left"  # "left" ou "right"


@dataclass
class TeamMember:
    """Membro de uma equipe com metadados"""
    entity: Entity
    role: TeamRole
    ai_controller: Optional[AIController] = None
    is_player_controlled: bool = False
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    total_healing_done: float = 0.0
    total_cc_applied: float = 0.0
    total_buffs_applied: int = 0
    kills: int = 0
    assists: int = 0
    deaths: int = 0


# ============================================================================
# COMPOSIÇÕES DE EQUIPE PRÉ-DEFINIDAS
# ============================================================================

TEAM_COMPOSITIONS = {
    # Composições 2v2
    "2v2_standard": [
        {"class": "warrior", "weapon": "sword", "role": TeamRole.DPS_MELEE},
        {"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER},
    ],
    "2v2_aggressive": [
        {"class": "berserker", "weapon": "greatsword", "role": TeamRole.DPS_MELEE},
        {"class": "assassin", "weapon": "dagger", "role": TeamRole.DPS_MELEE},
    ],
    "2v2_defensive": [
        {"class": "guardian", "weapon": "shield_bash", "role": TeamRole.TANK},
        {"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER},
    ],
    "2v2_control": [
        {"class": "controller", "weapon": "warhammer", "role": TeamRole.CONTROLLER},
        {"class": "trapper", "weapon": "trap_launcher", "role": TeamRole.CONTROLLER},
    ],
    
    # Composições 3v3
    "3v3_balanced": [
        {"class": "guardian", "weapon": "shield_bash", "role": TeamRole.TANK},
        {"class": "warrior", "weapon": "sword", "role": TeamRole.DPS_MELEE},
        {"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER},
    ],
    "3v3_aggressive": [
        {"class": "berserker", "weapon": "greatsword", "role": TeamRole.DPS_MELEE},
        {"class": "assassin", "weapon": "dagger", "role": TeamRole.DPS_MELEE},
        {"class": "ranger", "weapon": "bow", "role": TeamRole.DPS_RANGED},
    ],
    "3v3_control": [
        {"class": "controller", "weapon": "warhammer", "role": TeamRole.CONTROLLER},
        {"class": "trapper", "weapon": "trap_launcher", "role": TeamRole.CONTROLLER},
        {"class": "enchanter", "weapon": "tome", "role": TeamRole.SUPPORT},
    ],
    
    # Composições 5v5 (meta completo)
    "5v5_meta": [
        {"class": "guardian", "weapon": "shield_bash", "role": TeamRole.TANK},
        {"class": "berserker", "weapon": "greatsword", "role": TeamRole.DPS_MELEE},
        {"class": "ranger", "weapon": "bow", "role": TeamRole.DPS_RANGED},
        {"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER},
        {"class": "controller", "weapon": "warhammer", "role": TeamRole.CONTROLLER},
    ],
    "5v5_siege": [
        {"class": "guardian", "weapon": "shield_bash", "role": TeamRole.TANK},
        {"class": "tank", "weapon": "spear", "role": TeamRole.TANK},
        {"class": "ranger", "weapon": "bow", "role": TeamRole.DPS_RANGED},
        {"class": "enchanter", "weapon": "tome", "role": TeamRole.SUPPORT},
        {"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER},
    ],
    "5v5_dive": [
        {"class": "assassin", "weapon": "dagger", "role": TeamRole.DPS_MELEE},
        {"class": "berserker", "weapon": "greatsword", "role": TeamRole.DPS_MELEE},
        {"class": "lancer", "weapon": "spear", "role": TeamRole.DPS_MELEE},
        {"class": "controller", "weapon": "warhammer", "role": TeamRole.CONTROLLER},
        {"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER},
    ],
}


# ============================================================================
# FUNÇÕES DE RECOMPENSA ESPECIALIZADAS
# ============================================================================

class RewardCalculator:
    """Calculador de recompensas especializado por papel"""
    
    @staticmethod
    def calculate_healer_reward(member: TeamMember, team: List[TeamMember],
                                 enemies: List[TeamMember], dt: float) -> float:
        """Recompensa para healers - prioriza manter equipe viva"""
        reward = 0.0
        
        # Grande recompensa por cura feita
        if member.total_healing_done > 0:
            reward += member.total_healing_done * 0.3
        
        # Recompensa por aliados vivos com boa vida
        for ally in team:
            if ally.entity.is_alive():
                health_pct = ally.entity.health / ally.entity.stats_manager.get_stats().max_health
                reward += health_pct * 0.02
                
                # Bônus extra por manter aliados feridos vivos
                if health_pct < 0.5:
                    reward += 0.05
        
        # Penalidade por morte de aliados
        for ally in team:
            if not ally.entity.is_alive():
                reward -= 0.5
        
        # Penalidade por estar muito perto de inimigos (healer deve manter distância)
        for enemy in enemies:
            if enemy.entity.is_alive():
                dist = math.sqrt(
                    (member.entity.x - enemy.entity.x)**2 +
                    (member.entity.y - enemy.entity.y)**2
                )
                if dist < 100:
                    reward -= 0.02
                elif dist > 150 and dist < 250:
                    reward += 0.01  # Distância ideal
        
        # Sobrevivência
        reward += 0.001
        
        return reward
    
    @staticmethod
    def calculate_tank_reward(member: TeamMember, team: List[TeamMember],
                               enemies: List[TeamMember], dt: float) -> float:
        """Recompensa para tanks - prioriza absorver dano e proteger"""
        reward = 0.0
        
        # Recompensa por dano absorvido (tank QUER levar dano)
        if member.total_damage_taken > 0:
            # Mas não muito de uma vez (deve gerenciar)
            dmg_this_frame = member.total_damage_taken
            reward += min(dmg_this_frame * 0.05, 0.5)
        
        # Recompensa por estar na frente (entre aliados e inimigos)
        tank_to_enemy_dist = float('inf')
        ally_to_enemy_dist = float('inf')
        
        for enemy in enemies:
            if enemy.entity.is_alive():
                dist = math.sqrt(
                    (member.entity.x - enemy.entity.x)**2 +
                    (member.entity.y - enemy.entity.y)**2
                )
                tank_to_enemy_dist = min(tank_to_enemy_dist, dist)
                
                for ally in team:
                    if ally != member and ally.entity.is_alive():
                        ally_dist = math.sqrt(
                            (ally.entity.x - enemy.entity.x)**2 +
                            (ally.entity.y - enemy.entity.y)**2
                        )
                        ally_to_enemy_dist = min(ally_to_enemy_dist, ally_dist)
        
        # Bônus se tank está mais perto do inimigo que aliados
        if tank_to_enemy_dist < ally_to_enemy_dist:
            reward += 0.03
        
        # Recompensa por ter escudo ativo (Guardian)
        if hasattr(member.entity, 'shield') and member.entity.shield > 0:
            reward += 0.02
        
        # Bônus por aplicar CC (taunts, stuns via Shield Bash)
        if member.total_cc_applied > 0:
            reward += member.total_cc_applied * 0.1
        
        # Penalidade por morte
        if not member.entity.is_alive():
            reward -= 2.0
        
        reward += 0.001
        return reward
    
    @staticmethod
    def calculate_dps_reward(member: TeamMember, team: List[TeamMember],
                              enemies: List[TeamMember], dt: float) -> float:
        """Recompensa para DPS - prioriza causar dano"""
        reward = 0.0
        
        # Grande recompensa por dano causado
        if member.total_damage_dealt > 0:
            reward += member.total_damage_dealt * 0.15
        
        # Bônus por kill
        if member.kills > 0:
            reward += member.kills * 2.0
        
        # Bônus por assist
        if member.assists > 0:
            reward += member.assists * 0.5
        
        # Recompensa por focar inimigos com pouca vida
        for enemy in enemies:
            if enemy.entity.is_alive():
                health_pct = enemy.entity.health / enemy.entity.stats_manager.get_stats().max_health
                if health_pct < 0.3:
                    # Está perto de inimigo com pouca vida? Bom!
                    dist = math.sqrt(
                        (member.entity.x - enemy.entity.x)**2 +
                        (member.entity.y - enemy.entity.y)**2
                    )
                    if dist < 150:
                        reward += 0.05
        
        # Penalidade leve por morrer (DPS pode arriscar mais)
        if not member.entity.is_alive():
            reward -= 1.0
        
        # Penalidade leve por dano recebido
        if member.total_damage_taken > 0:
            reward -= member.total_damage_taken * 0.02
        
        reward += 0.001
        return reward
    
    @staticmethod
    def calculate_controller_reward(member: TeamMember, team: List[TeamMember],
                                     enemies: List[TeamMember], dt: float) -> float:
        """Recompensa para controllers - prioriza CC e controle de área"""
        reward = 0.0
        
        # Grande recompensa por CC aplicado
        if member.total_cc_applied > 0:
            reward += member.total_cc_applied * 0.25
        
        # Verificar inimigos com status effects
        cc_types = [StatusEffectType.STUN, StatusEffectType.ROOT, 
                    StatusEffectType.SLOW, StatusEffectType.SILENCE]
        
        for enemy in enemies:
            if enemy.entity.is_alive():
                for cc_type in cc_types:
                    if enemy.entity.status_effects.has_effect(cc_type):
                        reward += 0.05
        
        # Bônus por controlar múltiplos inimigos
        controlled_count = sum(
            1 for enemy in enemies 
            if enemy.entity.is_alive() and any(
                enemy.entity.status_effects.has_effect(cc) for cc in cc_types
            )
        )
        if controlled_count >= 2:
            reward += 0.1 * controlled_count
        
        # Recompensa por posicionamento (perto de múltiplos inimigos para AoE)
        enemies_nearby = sum(
            1 for enemy in enemies
            if enemy.entity.is_alive() and math.sqrt(
                (member.entity.x - enemy.entity.x)**2 +
                (member.entity.y - enemy.entity.y)**2
            ) < 150
        )
        if enemies_nearby >= 2:
            reward += 0.03 * enemies_nearby
        
        # Penalidade por morrer
        if not member.entity.is_alive():
            reward -= 1.5
        
        reward += 0.001
        return reward
    
    @staticmethod
    def calculate_support_reward(member: TeamMember, team: List[TeamMember],
                                  enemies: List[TeamMember], dt: float) -> float:
        """Recompensa para supports - prioriza buffs e utilidade"""
        reward = 0.0
        
        # Recompensa por buffs aplicados
        if member.total_buffs_applied > 0:
            reward += member.total_buffs_applied * 0.3
        
        # Verificar aliados com buffs ativos
        buff_types = [StatusEffectType.BUFF_ATTACK, StatusEffectType.BUFF_DEFENSE]
        
        for ally in team:
            if ally.entity.is_alive() and ally != member:
                for buff_type in buff_types:
                    if ally.entity.status_effects.has_effect(buff_type):
                        reward += 0.03
        
        # Bônus por posicionamento central (alcançar todos aliados)
        avg_team_x = sum(a.entity.x for a in team if a.entity.is_alive()) / max(1, sum(1 for a in team if a.entity.is_alive()))
        avg_team_y = sum(a.entity.y for a in team if a.entity.is_alive()) / max(1, sum(1 for a in team if a.entity.is_alive()))
        
        dist_to_center = math.sqrt(
            (member.entity.x - avg_team_x)**2 +
            (member.entity.y - avg_team_y)**2
        )
        if dist_to_center < 150:
            reward += 0.02
        
        # Penalidade por morrer
        if not member.entity.is_alive():
            reward -= 1.5
        
        reward += 0.001
        return reward
    
    @staticmethod
    def get_reward_function(role: TeamRole):
        """Retorna a função de recompensa apropriada para o papel"""
        reward_funcs = {
            TeamRole.HEALER: RewardCalculator.calculate_healer_reward,
            TeamRole.TANK: RewardCalculator.calculate_tank_reward,
            TeamRole.DPS_MELEE: RewardCalculator.calculate_dps_reward,
            TeamRole.DPS_RANGED: RewardCalculator.calculate_dps_reward,
            TeamRole.CONTROLLER: RewardCalculator.calculate_controller_reward,
            TeamRole.SUPPORT: RewardCalculator.calculate_support_reward,
        }
        return reward_funcs.get(role, RewardCalculator.calculate_dps_reward)


# ============================================================================
# AMBIENTE DE TREINAMENTO EM GRUPO
# ============================================================================

class TeamBattleEnv:
    """
    Ambiente de treinamento para combate em equipe.
    Suporta 2v2, 3v3, 5v5.
    """
    
    def __init__(self,
                 team_size: int = 2,
                 blue_team_config: Optional[List[Dict]] = None,
                 red_team_config: Optional[List[Dict]] = None,
                 training_mode: TrainingMode = TrainingMode.STANDARD,
                 training_role: Optional[TeamRole] = None,  # Papel sendo treinado
                 render_mode: str = None,
                 max_steps: int = 5000,
                 arena_size: Tuple[int, int] = (1200, 800)):
        """
        Args:
            team_size: Tamanho das equipes (2, 3, ou 5)
            blue_team_config: Configuração do time azul (agentes treinados)
            red_team_config: Configuração do time vermelho (oponentes)
            training_mode: Modo de treinamento especializado
            training_role: Papel específico sendo treinado (para observação/recompensa)
            render_mode: "human" para visualizar
            max_steps: Passos máximos por episódio
            arena_size: Tamanho da arena (maior para mais jogadores)
        """
        self.team_size = team_size
        self.training_mode = training_mode
        self.training_role = training_role
        self.render_mode = render_mode
        self.max_steps = max_steps
        
        # Configuração da arena (maior para equipes)
        self.arena_config = ArenaConfig(arena_size[0], arena_size[1], 50)
        self.game_config = GameConfig(max_episode_steps=max_steps)
        
        # Física
        self.physics = Physics(self.arena_config.width, self.arena_config.height)
        
        # Equipes
        self.blue_team: List[TeamMember] = []
        self.red_team: List[TeamMember] = []
        
        # Configurações de equipe
        self.blue_config = blue_team_config or self._get_default_config(team_size)
        self.red_config = red_team_config or self._get_default_config(team_size)
        
        # Estado
        self.steps = 0
        self.game_state: Optional[GameState] = None
        
        # Pygame
        self.screen = None
        self.clock = None
        if render_mode == "human":
            self._init_pygame()
        
        # Espaços
        self.observation_size = self._get_observation_size()
        self.action_size = 4  # move_x, move_y, attack, ability
        
        # Métricas
        self.episode_rewards = []
        self.blue_wins = 0
        self.red_wins = 0
        self.draws = 0
    
    def _get_default_config(self, team_size: int) -> List[Dict]:
        """Retorna configuração padrão baseada no tamanho da equipe"""
        if team_size == 2:
            return TEAM_COMPOSITIONS["2v2_standard"]
        elif team_size == 3:
            return TEAM_COMPOSITIONS["3v3_balanced"]
        else:
            return TEAM_COMPOSITIONS["5v5_meta"]
    
    def _init_pygame(self):
        """Inicializa pygame"""
        import pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.arena_config.width, self.arena_config.height)
        )
        pygame.display.set_caption(f"Circle Warriors - Team Battle {self.team_size}v{self.team_size}")
        self.clock = pygame.time.Clock()
    
    def _get_observation_size(self) -> int:
        """
        Tamanho do vetor de observação.
        Inclui: estado próprio + estado de cada aliado + estado de cada inimigo
        """
        # Estado próprio: 12 valores
        self_size = 12
        
        # Estado de aliados (team_size - 1): 10 valores cada
        allies_size = (self.team_size - 1) * 10
        
        # Estado de inimigos (team_size): 10 valores cada
        enemies_size = self.team_size * 10
        
        # Info global: 4 valores
        global_size = 4
        
        return self_size + allies_size + enemies_size + global_size
    
    def reset(self, seed: int = None) -> Tuple[List[np.ndarray], Dict]:
        """
        Reseta o ambiente.
        
        Returns:
            observations: Lista de observações (uma por agente azul)
            info: Informações adicionais
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        self.steps = 0
        self.game_state = GameState(self.arena_config, self.game_config)
        
        # Limpar equipes
        self.blue_team.clear()
        self.red_team.clear()
        
        # Criar time azul (esquerda)
        self._create_team(
            self.blue_config,
            team_list=self.blue_team,
            color=(100, 150, 255),
            spawn_side="left",
            is_blue=True
        )
        
        # Criar time vermelho (direita)
        self._create_team(
            self.red_config,
            team_list=self.red_team,
            color=(255, 100, 100),
            spawn_side="right",
            is_blue=False
        )
        
        # Configurar IAs para time vermelho (oponentes)
        for member in self.red_team:
            self._setup_ai(member, self.blue_team)
        
        # Reset game state
        self.game_state.reset()
        
        # Observações para cada agente azul
        observations = [self._get_observation(i) for i in range(len(self.blue_team))]
        
        info = {
            "blue_alive": sum(1 for m in self.blue_team if m.entity.is_alive()),
            "red_alive": sum(1 for m in self.red_team if m.entity.is_alive()),
        }
        
        return observations, info
    
    def _create_team(self, config: List[Dict], team_list: List[TeamMember],
                     color: Tuple[int, int, int], spawn_side: str, is_blue: bool):
        """Cria uma equipe baseada na configuração"""
        margin = 100
        
        if spawn_side == "left":
            x_range = (margin, self.arena_config.width // 3)
        else:
            x_range = (2 * self.arena_config.width // 3, self.arena_config.width - margin)
        
        y_spacing = (self.arena_config.height - 2 * margin) // max(1, len(config) - 1) if len(config) > 1 else 0
        
        for i, member_config in enumerate(config):
            # Posição
            x = random.randint(x_range[0], x_range[1])
            y = margin + i * y_spacing if len(config) > 1 else self.arena_config.height // 2
            y = min(max(y, margin), self.arena_config.height - margin)
            
            # Criar entidade
            entity = ClassRegistry.create(
                member_config["class"], x, y, color
            )
            entity.set_weapon(member_config["weapon"])
            entity._prev_health = entity.health
            
            # Criar membro
            role = member_config.get("role", TeamRole.DPS_MELEE)
            member = TeamMember(
                entity=entity,
                role=role,
                is_player_controlled=is_blue  # Blue team é controlado pela NN
            )
            
            team_list.append(member)
            self.game_state.add_entity(entity)
    
    def _setup_ai(self, member: TeamMember, targets: List[TeamMember]):
        """Configura IA para um membro da equipe"""
        class_name = member.entity.__class__.__name__
        weapon_name = member.entity.weapon.__class__.__name__ if member.entity.weapon else "Sword"
        
        ai = StrategicAI(class_name=class_name, weapon_name=weapon_name)
        ai.entity = member.entity
        ai.set_targets([t.entity for t in targets])
        
        member.ai_controller = ai
    
    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict]:
        """
        Executa um passo.
        
        Args:
            actions: Lista de ações (uma por agente azul)
        
        Returns:
            observations: Novas observações
            rewards: Recompensas por agente
            terminated: Se terminou
            truncated: Se foi cortado
            info: Informações
        """
        self.steps += 1
        dt = 1/60
        
        # Salvar estados anteriores
        prev_states = {}
        for team in [self.blue_team, self.red_team]:
            for member in team:
                prev_states[id(member)] = {
                    'health': member.entity.health,
                    'alive': member.entity.is_alive()
                }
        
        # Aplicar ações do time azul
        for i, (member, action) in enumerate(zip(self.blue_team, actions)):
            if member.entity.is_alive():
                self._apply_action(member.entity, action)
        
        # Atualizar IAs do time vermelho
        for member in self.red_team:
            if member.entity.is_alive() and member.ai_controller:
                # Atualizar alvos para aliados vivos
                member.ai_controller.set_targets([
                    m.entity for m in self.blue_team if m.entity.is_alive()
                ])
                member.ai_controller.update(dt)
        
        # Atualizar todas entidades
        all_entities = [m.entity for m in self.blue_team + self.red_team]
        for entity in all_entities:
            if entity.is_alive():
                entity.update(dt)
        
        # Física
        self.physics.handle_collisions(all_entities)
        
        # Limitar à arena
        import pygame
        arena_rect = pygame.Rect(*self.arena_config.playable_rect)
        for entity in all_entities:
            self.physics.constrain_to_arena(entity, arena_rect)
        
        # Atualizar métricas
        self._update_metrics(prev_states)
        
        # Calcular recompensas
        rewards = self._calculate_rewards(prev_states)
        
        # Verificar término
        blue_alive = sum(1 for m in self.blue_team if m.entity.is_alive())
        red_alive = sum(1 for m in self.red_team if m.entity.is_alive())
        
        terminated = False
        truncated = False
        
        if blue_alive == 0:
            terminated = True
            self.red_wins += 1
            # Penalidade extra por perder
            rewards = [r - 5.0 for r in rewards]
        elif red_alive == 0:
            terminated = True
            self.blue_wins += 1
            # Bônus extra por vencer
            rewards = [r + 10.0 for r in rewards]
        elif self.steps >= self.max_steps:
            truncated = True
            self.draws += 1
        
        # Renderizar
        if self.render_mode == "human":
            self.render()
        
        # Observações
        observations = [self._get_observation(i) for i in range(len(self.blue_team))]
        
        info = {
            "blue_alive": blue_alive,
            "red_alive": red_alive,
            "steps": self.steps,
            "blue_wins": self.blue_wins,
            "red_wins": self.red_wins,
            "draws": self.draws
        }
        
        return observations, rewards, terminated, truncated, info
    
    def _apply_action(self, entity: Entity, action: np.ndarray):
        """Aplica ação a uma entidade"""
        move_x = float(np.clip(action[0], -1, 1))
        move_y = float(np.clip(action[1], -1, 1))
        
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            entity.move(move_x, move_y)
        else:
            entity.moving = False
        
        if action[2] > 0.5:
            entity.attack()
        
        if action[3] > 0.5:
            entity.use_ability()
    
    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """Retorna observação para um agente específico"""
        obs = []
        agent = self.blue_team[agent_idx]
        
        # Estado próprio (12 valores)
        obs.extend([
            agent.entity.x / self.arena_config.width,
            agent.entity.y / self.arena_config.height,
            agent.entity.vx / 20,
            agent.entity.vy / 20,
            agent.entity.facing_angle / math.pi,
            agent.entity.health / agent.entity.stats_manager.get_stats().max_health,
            1.0 if agent.entity.is_alive() else 0.0,
            1.0 if agent.entity.invulnerable_time > 0 else 0.0,
            1.0 if agent.entity.weapon and agent.entity.weapon.can_attack else 0.0,
            1.0 if agent.entity.ability_cooldown <= 0 else 0.0,
            float(agent.role.value == "healer"),  # Info do papel
            float(agent.role.value == "tank"),
        ])
        
        # Estado dos aliados (10 valores cada)
        for i, ally in enumerate(self.blue_team):
            if i != agent_idx:
                if ally.entity.is_alive():
                    rel_x = (ally.entity.x - agent.entity.x) / self.arena_config.width
                    rel_y = (ally.entity.y - agent.entity.y) / self.arena_config.height
                    obs.extend([
                        rel_x,
                        rel_y,
                        math.sqrt(rel_x**2 + rel_y**2),
                        ally.entity.health / ally.entity.stats_manager.get_stats().max_health,
                        1.0,  # Vivo
                        ally.entity.vx / 20,
                        ally.entity.vy / 20,
                        float(ally.role.value == "healer"),
                        float(ally.role.value == "tank"),
                        float(ally.role.value == "dps_melee" or ally.role.value == "dps_ranged"),
                    ])
                else:
                    obs.extend([0.0] * 10)  # Morto
        
        # Estado dos inimigos (10 valores cada)
        for enemy in self.red_team:
            if enemy.entity.is_alive():
                rel_x = (enemy.entity.x - agent.entity.x) / self.arena_config.width
                rel_y = (enemy.entity.y - agent.entity.y) / self.arena_config.height
                obs.extend([
                    rel_x,
                    rel_y,
                    math.sqrt(rel_x**2 + rel_y**2),
                    enemy.entity.health / enemy.entity.stats_manager.get_stats().max_health,
                    1.0,  # Vivo
                    enemy.entity.vx / 20,
                    enemy.entity.vy / 20,
                    1.0 if enemy.entity.weapon and enemy.entity.weapon.is_attacking else 0.0,
                    float(enemy.role.value == "healer"),  # Priorizar healers
                    float(enemy.role.value == "dps_melee" or enemy.role.value == "dps_ranged"),
                ])
            else:
                obs.extend([0.0] * 10)  # Morto
        
        # Info global (4 valores)
        blue_alive = sum(1 for m in self.blue_team if m.entity.is_alive())
        red_alive = sum(1 for m in self.red_team if m.entity.is_alive())
        obs.extend([
            blue_alive / self.team_size,
            red_alive / self.team_size,
            self.steps / self.max_steps,
            (blue_alive - red_alive) / self.team_size,  # Vantagem numérica
        ])
        
        return np.array(obs, dtype=np.float32)
    
    def _update_metrics(self, prev_states: Dict):
        """Atualiza métricas de cada membro"""
        for team in [self.blue_team, self.red_team]:
            for member in team:
                prev = prev_states[id(member)]
                
                # Dano recebido
                if member.entity.health < prev['health']:
                    member.total_damage_taken += prev['health'] - member.entity.health
                
                # Morte
                if prev['alive'] and not member.entity.is_alive():
                    member.deaths += 1
    
    def _calculate_rewards(self, prev_states: Dict) -> List[float]:
        """Calcula recompensas baseadas no modo de treinamento"""
        rewards = []
        
        for member in self.blue_team:
            if self.training_mode == TrainingMode.STANDARD:
                reward = self._standard_reward(member, prev_states)
            else:
                # Usa função de recompensa especializada
                reward_func = RewardCalculator.get_reward_function(member.role)
                reward = reward_func(member, self.blue_team, self.red_team, 1/60)
            
            rewards.append(reward)
        
        return rewards
    
    def _standard_reward(self, member: TeamMember, prev_states: Dict) -> float:
        """Recompensa padrão (maximizar vitória)"""
        reward = 0.0
        
        prev = prev_states[id(member)]
        
        # Dano causado aos inimigos
        for enemy in self.red_team:
            enemy_prev = prev_states[id(enemy)]
            damage = enemy_prev['health'] - enemy.entity.health
            if damage > 0:
                reward += damage * 0.1
                member.total_damage_dealt += damage
        
        # Dano recebido
        damage_taken = prev['health'] - member.entity.health
        if damage_taken > 0:
            reward -= damage_taken * 0.03
        
        # Sobrevivência
        if member.entity.is_alive():
            reward += 0.001
        
        # Morte
        if prev['alive'] and not member.entity.is_alive():
            reward -= 3.0
        
        # Kill
        for enemy in self.red_team:
            enemy_prev = prev_states[id(enemy)]
            if enemy_prev['alive'] and not enemy.entity.is_alive():
                reward += 2.0
                member.kills += 1
        
        return reward
    
    def render(self):
        """Renderiza o jogo"""
        if self.screen is None:
            return
        
        import pygame
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
        
        # Fundo
        self.screen.fill((20, 20, 30))
        
        # Arena
        arena_rect = pygame.Rect(*self.arena_config.playable_rect)
        pygame.draw.rect(self.screen, (40, 40, 50), arena_rect)
        pygame.draw.rect(self.screen, (100, 100, 100), arena_rect, 2)
        
        # Linha central
        pygame.draw.line(
            self.screen,
            (60, 60, 70),
            (self.arena_config.width // 2, self.arena_config.border),
            (self.arena_config.width // 2, self.arena_config.height - self.arena_config.border),
            2
        )
        
        # Entidades
        for member in self.blue_team + self.red_team:
            if member.entity.is_alive():
                member.entity.draw(self.screen)
        
        # UI
        font = pygame.font.Font(None, 24)
        
        # Info azul
        blue_text = f"BLUE: {sum(1 for m in self.blue_team if m.entity.is_alive())}/{self.team_size}"
        text_surf = font.render(blue_text, True, (100, 150, 255))
        self.screen.blit(text_surf, (10, 10))
        
        # Info vermelho
        red_text = f"RED: {sum(1 for m in self.red_team if m.entity.is_alive())}/{self.team_size}"
        text_surf = font.render(red_text, True, (255, 100, 100))
        self.screen.blit(text_surf, (self.arena_config.width - 100, 10))
        
        # Steps
        step_text = f"Step: {self.steps}/{self.max_steps}"
        text_surf = font.render(step_text, True, (200, 200, 200))
        self.screen.blit(text_surf, (self.arena_config.width // 2 - 50, 10))
        
        # Placar
        score_text = f"W: {self.blue_wins} | L: {self.red_wins} | D: {self.draws}"
        text_surf = font.render(score_text, True, (200, 200, 200))
        self.screen.blit(text_surf, (10, self.arena_config.height - 30))
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """Fecha o ambiente"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None


# ============================================================================
# TREINADOR ESPECIALIZADO POR PAPEL
# ============================================================================

class RoleSpecificTrainer:
    """
    Treinador que treina agentes especificamente para seus papéis.
    Cada papel tem observações e recompensas otimizadas.
    """
    
    def __init__(self, 
                 role: TeamRole,
                 class_name: str,
                 weapon_name: str,
                 team_size: int = 2):
        """
        Args:
            role: Papel a ser treinado
            class_name: Classe do agente
            weapon_name: Arma do agente
            team_size: Tamanho da equipe
        """
        self.role = role
        self.class_name = class_name
        self.weapon_name = weapon_name
        self.team_size = team_size
        
        # Configurar equipe baseada no papel
        self.team_config = self._create_team_for_role()
        
        # Ambiente
        self.env = TeamBattleEnv(
            team_size=team_size,
            blue_team_config=self.team_config,
            training_mode=self._get_training_mode(),
            training_role=role,
            render_mode=None
        )
    
    def _create_team_for_role(self) -> List[Dict]:
        """Cria composição de equipe focada no papel sendo treinado"""
        config = [{"class": self.class_name, "weapon": self.weapon_name, "role": self.role}]
        
        # Adicionar aliados complementares
        if self.role == TeamRole.HEALER:
            # Healer precisa de aliados para curar
            config.append({"class": "warrior", "weapon": "sword", "role": TeamRole.DPS_MELEE})
            if self.team_size >= 3:
                config.append({"class": "guardian", "weapon": "shield_bash", "role": TeamRole.TANK})
        
        elif self.role == TeamRole.TANK:
            # Tank precisa de damage dealers para proteger
            config.append({"class": "ranger", "weapon": "bow", "role": TeamRole.DPS_RANGED})
            if self.team_size >= 3:
                config.append({"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER})
        
        elif self.role in [TeamRole.DPS_MELEE, TeamRole.DPS_RANGED]:
            # DPS precisa de suporte
            config.append({"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER})
            if self.team_size >= 3:
                config.append({"class": "guardian", "weapon": "shield_bash", "role": TeamRole.TANK})
        
        elif self.role == TeamRole.CONTROLLER:
            # Controller precisa de follow-up damage
            config.append({"class": "berserker", "weapon": "greatsword", "role": TeamRole.DPS_MELEE})
            if self.team_size >= 3:
                config.append({"class": "cleric", "weapon": "staff", "role": TeamRole.HEALER})
        
        elif self.role == TeamRole.SUPPORT:
            # Support precisa de aliados para buffar
            config.append({"class": "berserker", "weapon": "greatsword", "role": TeamRole.DPS_MELEE})
            if self.team_size >= 3:
                config.append({"class": "ranger", "weapon": "bow", "role": TeamRole.DPS_RANGED})
        
        return config[:self.team_size]
    
    def _get_training_mode(self) -> TrainingMode:
        """Retorna modo de treinamento para o papel"""
        mode_map = {
            TeamRole.HEALER: TrainingMode.HEALER_FOCUSED,
            TeamRole.TANK: TrainingMode.TANK_FOCUSED,
            TeamRole.DPS_MELEE: TrainingMode.DPS_FOCUSED,
            TeamRole.DPS_RANGED: TrainingMode.DPS_FOCUSED,
            TeamRole.CONTROLLER: TrainingMode.CC_FOCUSED,
            TeamRole.SUPPORT: TrainingMode.SUPPORT_FOCUSED,
        }
        return mode_map.get(self.role, TrainingMode.STANDARD)
    
    def train(self, total_episodes: int = 1000, save_path: str = None):
        """Treina o agente para o papel específico"""
        print(f"\n{'='*60}")
        print(f"Treinando {self.role.value.upper()}: {self.class_name} + {self.weapon_name}")
        print(f"Modo: {self._get_training_mode().value}")
        print(f"{'='*60}\n")
        
        # Aqui entraria o código de treinamento com PPO
        # Por enquanto, placeholder
        
        if save_path:
            print(f"Modelo seria salvo em: {save_path}")


# ============================================================================
# SISTEMA DE TORNEIO
# ============================================================================

class Tournament:
    """
    Sistema de torneio entre equipes/modelos treinados.
    """
    
    def __init__(self, team_size: int = 2):
        self.team_size = team_size
        self.participants: List[Dict] = []
        self.results: List[Dict] = []
    
    def register_team(self, name: str, config: List[Dict], model_paths: List[str] = None):
        """Registra uma equipe no torneio"""
        self.participants.append({
            "name": name,
            "config": config,
            "model_paths": model_paths,
            "wins": 0,
            "losses": 0,
            "points": 0
        })
    
    def run_match(self, team1_idx: int, team2_idx: int, 
                  num_rounds: int = 3, render: bool = False) -> Dict:
        """Executa uma partida entre duas equipes"""
        team1 = self.participants[team1_idx]
        team2 = self.participants[team2_idx]
        
        env = TeamBattleEnv(
            team_size=self.team_size,
            blue_team_config=team1["config"],
            red_team_config=team2["config"],
            render_mode="human" if render else None
        )
        
        results = {"team1_wins": 0, "team2_wins": 0, "draws": 0}
        
        for round_num in range(num_rounds):
            obs, _ = env.reset()
            done = False
            
            while not done:
                # Ações aleatórias por enquanto (substituir por modelos)
                actions = [np.random.uniform(-1, 1, 4) for _ in range(self.team_size)]
                obs, rewards, terminated, truncated, info = env.step(actions)
                done = terminated or truncated
            
            if info["blue_alive"] > 0 and info["red_alive"] == 0:
                results["team1_wins"] += 1
            elif info["red_alive"] > 0 and info["blue_alive"] == 0:
                results["team2_wins"] += 1
            else:
                results["draws"] += 1
        
        env.close()
        
        # Atualizar pontos
        if results["team1_wins"] > results["team2_wins"]:
            team1["wins"] += 1
            team1["points"] += 3
            team2["losses"] += 1
        elif results["team2_wins"] > results["team1_wins"]:
            team2["wins"] += 1
            team2["points"] += 3
            team1["losses"] += 1
        else:
            team1["points"] += 1
            team2["points"] += 1
        
        match_result = {
            "team1": team1["name"],
            "team2": team2["name"],
            "results": results
        }
        self.results.append(match_result)
        
        return match_result
    
    def run_round_robin(self, num_rounds: int = 3, render: bool = False):
        """Executa todas as partidas (todos contra todos)"""
        print("\n" + "="*60)
        print("TORNEIO - ROUND ROBIN")
        print("="*60 + "\n")
        
        for i in range(len(self.participants)):
            for j in range(i + 1, len(self.participants)):
                result = self.run_match(i, j, num_rounds, render)
                print(f"{result['team1']} vs {result['team2']}: "
                      f"{result['results']['team1_wins']}-{result['results']['team2_wins']}")
        
        self.print_standings()
    
    def print_standings(self):
        """Mostra classificação"""
        sorted_teams = sorted(self.participants, key=lambda x: (-x["points"], -x["wins"]))
        
        print("\n" + "="*60)
        print("CLASSIFICAÇÃO FINAL")
        print("="*60)
        print(f"{'Pos':<5}{'Equipe':<20}{'V':<5}{'D':<5}{'Pts':<5}")
        print("-"*40)
        
        for i, team in enumerate(sorted_teams, 1):
            print(f"{i:<5}{team['name']:<20}{team['wins']:<5}{team['losses']:<5}{team['points']:<5}")


# ============================================================================
# MAIN - DEMONSTRAÇÃO
# ============================================================================

def demo_team_battle():
    """Demonstração do sistema de batalha em equipe"""
    print("\n" + "="*60)
    print("DEMO: Batalha em Equipe 2v2")
    print("="*60 + "\n")
    
    env = TeamBattleEnv(
        team_size=2,
        blue_team_config=TEAM_COMPOSITIONS["2v2_standard"],
        red_team_config=TEAM_COMPOSITIONS["2v2_aggressive"],
        render_mode="human"
    )
    
    obs, info = env.reset()
    print(f"Observação shape: {obs[0].shape}")
    print(f"Blue alive: {info['blue_alive']}, Red alive: {info['red_alive']}")
    
    done = False
    total_reward = [0.0, 0.0]
    
    while not done:
        # Ações aleatórias para demonstração
        actions = [np.random.uniform(-1, 1, 4) for _ in range(2)]
        obs, rewards, terminated, truncated, info = env.step(actions)
        
        total_reward[0] += rewards[0]
        total_reward[1] += rewards[1]
        done = terminated or truncated
    
    print(f"\nResultado: Blue {info['blue_alive']} vs Red {info['red_alive']}")
    print(f"Recompensas totais: {total_reward}")
    
    env.close()


def demo_tournament():
    """Demonstração do sistema de torneio"""
    print("\n" + "="*60)
    print("DEMO: Mini Torneio 2v2")
    print("="*60 + "\n")
    
    tournament = Tournament(team_size=2)
    
    # Registrar equipes
    tournament.register_team("Standard", TEAM_COMPOSITIONS["2v2_standard"])
    tournament.register_team("Aggressive", TEAM_COMPOSITIONS["2v2_aggressive"])
    tournament.register_team("Defensive", TEAM_COMPOSITIONS["2v2_defensive"])
    tournament.register_team("Control", TEAM_COMPOSITIONS["2v2_control"])
    
    # Executar torneio
    tournament.run_round_robin(num_rounds=1, render=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Sistema de Treinamento em Grupo")
    parser.add_argument("--demo", choices=["battle", "tournament"], default="battle",
                        help="Demonstração a executar")
    parser.add_argument("--team-size", type=int, default=2, choices=[2, 3, 5],
                        help="Tamanho das equipes")
    parser.add_argument("--render", action="store_true",
                        help="Visualizar partidas")
    
    args = parser.parse_args()
    
    if args.demo == "battle":
        demo_team_battle()
    elif args.demo == "tournament":
        demo_tournament()
