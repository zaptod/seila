"""
Sistema de Controle
===================
Interface para controle de entidades por jogadores ou IA.
Permite fácil integração com redes neurais.
"""

import pygame
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from entities import Entity


class Controller(ABC):
    """Classe base abstrata para controladores"""
    
    def __init__(self):
        self.entity: Optional['Entity'] = None
    
    @abstractmethod
    def update(self, dt: float):
        """Atualiza o controlador e processa ações"""
        pass
    
    @abstractmethod
    def get_actions(self) -> Dict:
        """Retorna as ações atuais do controlador"""
        pass


class PlayerController(Controller):
    """Controlador para jogadores humanos via teclado"""
    
    def __init__(self, key_up: int, key_down: int, key_left: int, key_right: int, 
                 key_attack: int, key_ability: int = None):
        super().__init__()
        self.key_up = key_up
        self.key_down = key_down
        self.key_left = key_left
        self.key_right = key_right
        self.key_attack = key_attack
        self.key_ability = key_ability
        
        # Estado das ações
        self.actions = {
            'move_x': 0,
            'move_y': 0,
            'attack': False,
            'ability': False
        }
    
    def update(self, dt: float):
        """Processa input do teclado"""
        if not self.entity:
            return
        
        keys = pygame.key.get_pressed()
        
        # Movimento
        dx, dy = 0, 0
        if keys[self.key_up]:
            dy -= 1
        if keys[self.key_down]:
            dy += 1
        if keys[self.key_left]:
            dx -= 1
        if keys[self.key_right]:
            dx += 1
        
        self.actions['move_x'] = dx
        self.actions['move_y'] = dy
        
        # Aplicar movimento
        if dx != 0 or dy != 0:
            self.entity.move(dx, dy)
        else:
            self.entity.moving = False
        
        # Ataque
        self.actions['attack'] = keys[self.key_attack]
        if self.actions['attack']:
            self.entity.attack()
        
        # Habilidade
        if self.key_ability:
            self.actions['ability'] = keys[self.key_ability]
            if self.actions['ability']:
                self.entity.use_ability()
    
    def get_actions(self) -> Dict:
        return self.actions.copy()


class AIController(Controller):
    """
    Controlador base para IA.
    Pode ser estendido para usar redes neurais.
    """
    
    # Definição do espaço de ações (para redes neurais)
    ACTION_SPACE = {
        'move_x': (-1, 1),      # Contínuo: -1 (esquerda) a 1 (direita)
        'move_y': (-1, 1),      # Contínuo: -1 (cima) a 1 (baixo)
        'attack': (0, 1),       # Discreto: 0 ou 1
        'ability': (0, 1)       # Discreto: 0 ou 1
    }
    
    def __init__(self):
        super().__init__()
        self.actions = {
            'move_x': 0,
            'move_y': 0,
            'attack': False,
            'ability': False
        }
        
        # Referência para outros combatentes (para observação)
        self.enemies: List['Entity'] = []
        self.allies: List['Entity'] = []
    
    def set_targets(self, enemies: List['Entity'], allies: List['Entity'] = None):
        """Define os alvos da IA"""
        self.enemies = enemies or []
        self.allies = allies or []
    
    @abstractmethod
    def decide_actions(self, observation: Dict) -> Dict:
        """
        Decide as ações baseado na observação.
        Este método deve ser implementado pela rede neural.
        
        Args:
            observation: Estado do jogo (output de get_observation)
        
        Returns:
            Dict com as ações a tomar
        """
        pass
    
    def _is_enemy_visible(self, enemy: 'Entity') -> bool:
        """Verifica se um inimigo é visível para a IA (não está invisível)"""
        # Verificar se o inimigo tem o atributo invisible (Assassino)
        if hasattr(enemy, 'invisible') and enemy.invisible:
            return False
        return True
    
    def get_observation(self) -> Dict:
        """
        Retorna a observação do estado do jogo para a IA.
        Este é o input para a rede neural.
        Inimigos invisíveis não são incluídos na observação.
        """
        if not self.entity:
            return {}
        
        # Filtrar apenas inimigos visíveis (não invisíveis)
        visible_enemies = [e for e in self.enemies if e.is_alive() and self._is_enemy_visible(e)]
        
        obs = {
            'self': self.entity.get_state(),
            'enemies': [e.get_state() for e in visible_enemies],
            'allies': [a.get_state() for a in self.allies if a.is_alive()]
        }
        
        return obs
    
    def get_observation_vector(self) -> np.ndarray:
        """
        Retorna a observação como um vetor numpy normalizado.
        Ideal para redes neurais.
        Inimigos invisíveis não são incluídos.
        """
        if not self.entity:
            return np.zeros(self.get_observation_size())
        
        obs = []
        
        # Estado próprio (normalizado)
        self_state = self.entity.get_state()
        obs.extend([
            self_state['x'] / 1200,  # Normalizado pela largura da tela
            self_state['y'] / 800,   # Normalizado pela altura da tela
            self_state['vx'] / 20,   # Normalizado por velocidade máxima aproximada
            self_state['vy'] / 20,
            self_state['facing_angle'] / math.pi,  # -1 a 1
            self_state['health_ratio'],
            1.0 if self_state['is_invulnerable'] else 0.0,
            1.0 if self_state['weapon']['can_attack'] else 0.0,
            1.0 if self_state['ability_ready'] else 0.0
        ])
        
        # Estado do inimigo mais próximo VISÍVEL (se houver)
        # Inimigos invisíveis não são detectados pela IA
        visible_enemies = [e for e in self.enemies if e.is_alive() and self._is_enemy_visible(e)]
        
        if visible_enemies:
            closest_enemy = min(
                visible_enemies,
                key=lambda e: math.sqrt((e.x - self.entity.x)**2 + (e.y - self.entity.y)**2),
                default=None
            )
            
            if closest_enemy:
                enemy_state = closest_enemy.get_state()
                # Posição relativa
                rel_x = (enemy_state['x'] - self_state['x']) / 1200
                rel_y = (enemy_state['y'] - self_state['y']) / 800
                distance = math.sqrt(rel_x**2 + rel_y**2)
                angle_to_enemy = math.atan2(rel_y, rel_x)
                
                obs.extend([
                    rel_x,
                    rel_y,
                    distance,
                    angle_to_enemy / math.pi,
                    enemy_state['vx'] / 20,
                    enemy_state['vy'] / 20,
                    enemy_state['health_ratio'],
                    1.0 if enemy_state['weapon']['is_attacking'] else 0.0
                ])
            else:
                obs.extend([0.0] * 8)
        else:
            obs.extend([0.0] * 8)
        
        return np.array(obs, dtype=np.float32)
    
    @staticmethod
    def get_observation_size() -> int:
        """Retorna o tamanho do vetor de observação"""
        return 9 + 8  # Self state + enemy state
    
    @staticmethod
    def get_action_size() -> int:
        """Retorna o tamanho do vetor de ações"""
        return 4  # move_x, move_y, attack, ability
    
    def actions_from_vector(self, action_vector: np.ndarray) -> Dict:
        """
        Converte um vetor de ações (output da rede neural) para dict de ações.
        
        Args:
            action_vector: Array com [move_x, move_y, attack, ability]
        
        Returns:
            Dict de ações
        """
        return {
            'move_x': float(np.clip(action_vector[0], -1, 1)),
            'move_y': float(np.clip(action_vector[1], -1, 1)),
            'attack': bool(action_vector[2] > 0.5),
            'ability': bool(action_vector[3] > 0.5)
        }
    
    def update(self, dt: float):
        """Atualiza a IA"""
        if not self.entity:
            return
        
        # Obter observação
        observation = self.get_observation()
        
        # Decidir ações (implementado pela subclasse)
        self.actions = self.decide_actions(observation)
        
        # Aplicar ações
        if self.actions['move_x'] != 0 or self.actions['move_y'] != 0:
            self.entity.move(self.actions['move_x'], self.actions['move_y'])
        else:
            self.entity.moving = False
        
        if self.actions['attack']:
            self.entity.attack()
        
        if self.actions['ability']:
            self.entity.use_ability()
    
    def get_actions(self) -> Dict:
        return self.actions.copy()


class SimpleAI(AIController):
    """
    IA simples baseada em regras.
    Útil para testes e como baseline.
    """
    
    def __init__(self, aggression: float = 0.7):
        super().__init__()
        self.aggression = aggression  # 0-1, quão agressiva é a IA
        self.preferred_distance = 60  # Distância preferida do alvo
    
    def decide_actions(self, observation: Dict) -> Dict:
        actions = {
            'move_x': 0,
            'move_y': 0,
            'attack': False,
            'ability': False
        }
        
        if not observation.get('enemies'):
            return actions
        
        self_state = observation['self']
        
        # Encontrar inimigo mais próximo
        closest_enemy = None
        min_distance = float('inf')
        
        for enemy in observation['enemies']:
            dx = enemy['x'] - self_state['x']
            dy = enemy['y'] - self_state['y']
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_distance:
                min_distance = dist
                closest_enemy = enemy
        
        if not closest_enemy:
            return actions
        
        # Calcular direção para o inimigo
        dx = closest_enemy['x'] - self_state['x']
        dy = closest_enemy['y'] - self_state['y']
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > 0:
            dx /= distance
            dy /= distance
        
        # Decidir movimento baseado na agressividade e distância
        if distance > self.preferred_distance + 20:
            # Aproximar
            actions['move_x'] = dx * self.aggression
            actions['move_y'] = dy * self.aggression
        elif distance < self.preferred_distance - 20:
            # Recuar um pouco (se não for muito agressivo)
            actions['move_x'] = -dx * (1 - self.aggression) * 0.5
            actions['move_y'] = -dy * (1 - self.aggression) * 0.5
        else:
            # Movimento lateral para cercar
            actions['move_x'] = -dy * 0.3
            actions['move_y'] = dx * 0.3
        
        # Atacar se estiver perto o suficiente e a arma estiver pronta
        weapon_state = self_state.get('weapon', {})
        if distance < 80 and weapon_state.get('can_attack', False):
            actions['attack'] = True
        
        # Usar habilidade se disponível e em boa situação
        if self_state['ability_ready']:
            # Usar habilidade se vida baixa ou inimigo perto
            if self_state['health_ratio'] < 0.3 or distance < 50:
                actions['ability'] = True
        
        return actions


# =============================================================================
# ESTRATÉGIAS ESPECÍFICAS POR CLASSE + ARMA
# =============================================================================

# Dicionário com todas as 20 estratégias (5 classes × 4 armas)
CLASS_WEAPON_STRATEGIES = {
    # =========================================================================
    # WARRIOR - Classe equilibrada, foco em combate sustentado
    # =========================================================================
    ('Warrior', 'Sword'): {
        'name': 'Balanced Fighter',
        'description': 'Combatente equilibrado, mantém distância média e ataca com consistência',
        'aggression': 0.6,
        'preferred_distance': 55,
        'attack_distance': 70,
        'retreat_health': 0.25,
        'ability_health_threshold': 0.4,  # Usa escudo quando vida < 40%
        'ability_distance_threshold': 80,  # Ou quando inimigo muito perto
        'strafe_intensity': 0.4,
        'approach_speed': 0.7,
        'retreat_speed': 0.5,
        'combo_tendency': 0.6,  # Tendência a fazer combos
    },
    ('Warrior', 'Greatsword'): {
        'name': 'Heavy Hitter',
        'description': 'Ataques lentos mas devastadores, aproxima com cautela',
        'aggression': 0.5,
        'preferred_distance': 65,
        'attack_distance': 85,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 60,
        'strafe_intensity': 0.3,
        'approach_speed': 0.5,
        'retreat_speed': 0.4,
        'combo_tendency': 0.3,  # Menos combos, ataques únicos fortes
    },
    ('Warrior', 'Dagger'): {
        'name': 'Swift Guardian',
        'description': 'Usa velocidade da adaga com proteção do escudo',
        'aggression': 0.7,
        'preferred_distance': 40,
        'attack_distance': 55,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.35,
        'ability_distance_threshold': 50,
        'strafe_intensity': 0.6,
        'approach_speed': 0.85,
        'retreat_speed': 0.6,
        'combo_tendency': 0.8,
    },
    ('Warrior', 'Spear'): {
        'name': 'Defensive Poker',
        'description': 'Mantém distância com lança, usa escudo para proteger',
        'aggression': 0.45,
        'preferred_distance': 85,
        'attack_distance': 110,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.45,
        'ability_distance_threshold': 70,
        'strafe_intensity': 0.5,
        'approach_speed': 0.5,
        'retreat_speed': 0.6,
        'combo_tendency': 0.4,
    },
    
    # =========================================================================
    # BERSERKER - Ultra agressivo, não recua, dano aumenta com vida baixa
    # =========================================================================
    ('Berserker', 'Sword'): {
        'name': 'Raging Blade',
        'description': 'Agressivo constante, busca combate corpo a corpo',
        'aggression': 0.85,
        'preferred_distance': 45,
        'attack_distance': 70,
        'retreat_health': 0.0,  # Nunca recua!
        'ability_health_threshold': 0.6,  # Usa rage cedo para buff de dano
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.25,
        'approach_speed': 0.9,
        'retreat_speed': 0.0,  # Não recua
        'combo_tendency': 0.75,
        'berserker_mode': True,  # Fica mais agressivo com vida baixa
    },
    ('Berserker', 'Greatsword'): {
        'name': 'Furious Executioner',
        'description': 'Devastação total, ataques massivos sem medo',
        'aggression': 0.95,
        'preferred_distance': 50,
        'attack_distance': 85,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.15,
        'approach_speed': 0.8,
        'retreat_speed': 0.0,
        'combo_tendency': 0.5,
        'berserker_mode': True,
    },
    ('Berserker', 'Dagger'): {
        'name': 'Frenzy Striker',
        'description': 'Ataque frenético, muitos golpes rápidos',
        'aggression': 0.9,
        'preferred_distance': 30,
        'attack_distance': 50,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 60,
        'strafe_intensity': 0.4,
        'approach_speed': 1.0,
        'retreat_speed': 0.0,
        'combo_tendency': 0.95,  # Máximo de combos
        'berserker_mode': True,
    },
    ('Berserker', 'Spear'): {
        'name': 'Charging Maniac',
        'description': 'Usa alcance da lança para pressionar constantemente',
        'aggression': 0.8,
        'preferred_distance': 70,
        'attack_distance': 110,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.55,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.2,
        'approach_speed': 0.85,
        'retreat_speed': 0.0,
        'combo_tendency': 0.6,
        'berserker_mode': True,
    },
    
    # =========================================================================
    # ASSASSIN - Hit and run, usa invisibilidade taticamente
    # =========================================================================
    ('Assassin', 'Sword'): {
        'name': 'Shadow Blade',
        'description': 'Aproxima invisível, ataca e recua',
        'aggression': 0.65,
        'preferred_distance': 50,
        'attack_distance': 70,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.8,  # Usa invisibilidade para aproximar
        'ability_distance_threshold': 150,  # Usa quando longe para aproximar
        'strafe_intensity': 0.7,
        'approach_speed': 0.8,
        'retreat_speed': 0.9,  # Recua rápido após ataque
        'combo_tendency': 0.5,
        'hit_and_run': True,  # Padrão hit and run
        'stealth_approach': True,  # Aproxima em stealth
    },
    ('Assassin', 'Greatsword'): {
        'name': 'Phantom Executioner',
        'description': 'Um golpe devastador do nada',
        'aggression': 0.55,
        'preferred_distance': 60,
        'attack_distance': 85,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.5,
        'approach_speed': 0.6,
        'retreat_speed': 0.8,
        'combo_tendency': 0.2,  # Um golpe forte e some
        'hit_and_run': True,
        'stealth_approach': True,
    },
    ('Assassin', 'Dagger'): {
        'name': 'Silent Death',
        'description': 'O combo perfeito: invisibilidade + adagas rápidas',
        'aggression': 0.75,
        'preferred_distance': 25,
        'attack_distance': 45,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.8,
        'approach_speed': 1.0,
        'retreat_speed': 1.0,
        'combo_tendency': 0.9,
        'hit_and_run': True,
        'stealth_approach': True,
    },
    ('Assassin', 'Spear'): {
        'name': 'Ghost Lancer',
        'description': 'Ataques surpresa de longa distância',
        'aggression': 0.5,
        'preferred_distance': 80,
        'attack_distance': 110,
        'retreat_health': 0.5,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 140,
        'strafe_intensity': 0.6,
        'approach_speed': 0.7,
        'retreat_speed': 0.85,
        'combo_tendency': 0.4,
        'hit_and_run': True,
        'stealth_approach': True,
    },
    
    # =========================================================================
    # TANK - Defensivo, absorve dano, contra-ataca
    # =========================================================================
    ('Tank', 'Sword'): {
        'name': 'Iron Bulwark',
        'description': 'Avança devagar, escudo sempre pronto',
        'aggression': 0.4,
        'preferred_distance': 50,
        'attack_distance': 70,
        'retreat_health': 0.15,  # Quase nunca recua
        'ability_health_threshold': 0.7,  # Usa escudo frequentemente
        'ability_distance_threshold': 60,
        'strafe_intensity': 0.2,
        'approach_speed': 0.4,
        'retreat_speed': 0.3,
        'combo_tendency': 0.5,
        'defensive_style': True,  # Espera o inimigo atacar primeiro
        'shield_reactive': True,  # Usa escudo em reação a ataques
    },
    ('Tank', 'Greatsword'): {
        'name': 'Unstoppable Force',
        'description': 'Lento mas imparável, golpes devastadores',
        'aggression': 0.35,
        'preferred_distance': 55,
        'attack_distance': 85,
        'retreat_health': 0.1,
        'ability_health_threshold': 0.6,
        'ability_distance_threshold': 70,
        'strafe_intensity': 0.15,
        'approach_speed': 0.35,
        'retreat_speed': 0.25,
        'combo_tendency': 0.3,
        'defensive_style': True,
        'shield_reactive': True,
    },
    ('Tank', 'Dagger'): {
        'name': 'Counter Striker',
        'description': 'Aguarda e contra-ataca rapidamente',
        'aggression': 0.5,
        'preferred_distance': 40,
        'attack_distance': 55,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.65,
        'ability_distance_threshold': 50,
        'strafe_intensity': 0.3,
        'approach_speed': 0.5,
        'retreat_speed': 0.35,
        'combo_tendency': 0.7,
        'defensive_style': True,
        'shield_reactive': True,
    },
    ('Tank', 'Spear'): {
        'name': 'Fortress Keeper',
        'description': 'Mantém distância máxima, pokes seguros',
        'aggression': 0.3,
        'preferred_distance': 95,
        'attack_distance': 115,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.25,
        'approach_speed': 0.3,
        'retreat_speed': 0.4,
        'combo_tendency': 0.35,
        'defensive_style': True,
        'shield_reactive': True,
    },
    
    # =========================================================================
    # LANCER - Mobilidade extrema, usa dash para entrar e sair
    # =========================================================================
    ('Lancer', 'Sword'): {
        'name': 'Blitz Striker',
        'description': 'Dash para entrar, ataca, dash para sair',
        'aggression': 0.7,
        'preferred_distance': 80,  # Fica longe, usa dash para entrar
        'attack_distance': 70,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.9,  # Usa dash muito
        'ability_distance_threshold': 100,  # Dash quando longe
        'strafe_intensity': 0.5,
        'approach_speed': 0.6,
        'retreat_speed': 0.8,
        'combo_tendency': 0.6,
        'dash_in': True,  # Usa dash para aproximar
        'dash_out': True,  # Usa dash para escapar
    },
    ('Lancer', 'Greatsword'): {
        'name': 'Diving Destroyer',
        'description': 'Mergulha com dash, desfere golpe pesado',
        'aggression': 0.6,
        'preferred_distance': 90,
        'attack_distance': 85,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 110,
        'strafe_intensity': 0.35,
        'approach_speed': 0.5,
        'retreat_speed': 0.7,
        'combo_tendency': 0.3,
        'dash_in': True,
        'dash_out': True,
    },
    ('Lancer', 'Dagger'): {
        'name': 'Flash Assassin',
        'description': 'Velocidade máxima, dash + adagas = morte',
        'aggression': 0.8,
        'preferred_distance': 70,
        'attack_distance': 50,
        'retreat_health': 0.25,
        'ability_health_threshold': 0.95,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.7,
        'approach_speed': 0.9,
        'retreat_speed': 0.95,
        'combo_tendency': 0.9,
        'dash_in': True,
        'dash_out': True,
    },
    ('Lancer', 'Spear'): {
        'name': 'Dragoon',
        'description': 'O combo clássico: mobilidade + alcance máximo',
        'aggression': 0.65,
        'preferred_distance': 100,
        'attack_distance': 115,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 130,
        'strafe_intensity': 0.45,
        'approach_speed': 0.65,
        'retreat_speed': 0.75,
        'combo_tendency': 0.5,
        'dash_in': True,
        'dash_out': True,
    },
    
    # =========================================================================
    # CLERIC - Curador, foca em manter aliados vivos
    # =========================================================================
    ('Cleric', 'Staff'): {
        'name': 'Divine Healer',
        'description': 'Curador puro, fica atrás curando aliados',
        'aggression': 0.2,
        'preferred_distance': 120,  # Fica longe dos inimigos
        'attack_distance': 100,
        'retreat_health': 0.6,  # Recua cedo
        'ability_health_threshold': 0.7,  # Usa cura em área quando aliados perdem vida
        'ability_distance_threshold': 200,  # Distância para curar aliados
        'strafe_intensity': 0.6,
        'approach_speed': 0.3,
        'retreat_speed': 0.8,
        'combo_tendency': 0.3,
        'healer_mode': True,
        'protect_allies': True,
        'ally_health_priority': 0.5,  # Prioriza aliados com menos de 50% vida
    },
    ('Cleric', 'Sword'): {
        'name': 'Battle Priest',
        'description': 'Curador que também luta na linha de frente',
        'aggression': 0.45,
        'preferred_distance': 60,
        'attack_distance': 70,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.6,
        'ability_distance_threshold': 150,
        'strafe_intensity': 0.4,
        'approach_speed': 0.5,
        'retreat_speed': 0.6,
        'combo_tendency': 0.5,
        'healer_mode': True,
        'protect_allies': True,
    },
    ('Cleric', 'Warhammer'): {
        'name': 'Templar',
        'description': 'Curador agressivo que stuna inimigos',
        'aggression': 0.5,
        'preferred_distance': 55,
        'attack_distance': 65,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.55,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.35,
        'approach_speed': 0.55,
        'retreat_speed': 0.5,
        'combo_tendency': 0.4,
        'healer_mode': True,
        'protect_allies': True,
        'stun_priority': True,
    },
    ('Cleric', 'Tome'): {
        'name': 'Archpriest',
        'description': 'Combina cura com buffs para aliados',
        'aggression': 0.25,
        'preferred_distance': 100,
        'attack_distance': 90,
        'retreat_health': 0.55,
        'ability_health_threshold': 0.65,
        'ability_distance_threshold': 180,
        'strafe_intensity': 0.5,
        'approach_speed': 0.35,
        'retreat_speed': 0.7,
        'combo_tendency': 0.3,
        'healer_mode': True,
        'protect_allies': True,
        'buff_priority': True,
    },
    
    # =========================================================================
    # GUARDIAN - Protetor, absorve dano e protege aliados
    # =========================================================================
    ('Guardian', 'Shield_bash'): {
        'name': 'Aegis Bearer',
        'description': 'Protetor máximo, usa escudo para proteger o time',
        'aggression': 0.35,
        'preferred_distance': 50,
        'attack_distance': 55,
        'retreat_health': 0.15,
        'ability_health_threshold': 0.8,  # Usa escudo de área frequentemente
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.2,
        'approach_speed': 0.4,
        'retreat_speed': 0.25,
        'combo_tendency': 0.4,
        'guardian_mode': True,
        'protect_allies': True,
        'body_block': True,  # Fica entre inimigo e aliado
    },
    ('Guardian', 'Sword'): {
        'name': 'Sentinel',
        'description': 'Protetor equilibrado entre ataque e defesa',
        'aggression': 0.4,
        'preferred_distance': 55,
        'attack_distance': 70,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 110,
        'strafe_intensity': 0.25,
        'approach_speed': 0.45,
        'retreat_speed': 0.3,
        'combo_tendency': 0.5,
        'guardian_mode': True,
        'protect_allies': True,
    },
    ('Guardian', 'Warhammer'): {
        'name': 'Protector',
        'description': 'Defende com escudos e stuna ameaças',
        'aggression': 0.45,
        'preferred_distance': 50,
        'attack_distance': 60,
        'retreat_health': 0.15,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.2,
        'approach_speed': 0.4,
        'retreat_speed': 0.25,
        'combo_tendency': 0.35,
        'guardian_mode': True,
        'protect_allies': True,
        'stun_priority': True,
    },
    ('Guardian', 'Spear'): {
        'name': 'Phalanx',
        'description': 'Protege de longe, poke e escudo',
        'aggression': 0.3,
        'preferred_distance': 80,
        'attack_distance': 100,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.3,
        'approach_speed': 0.35,
        'retreat_speed': 0.35,
        'combo_tendency': 0.3,
        'guardian_mode': True,
        'protect_allies': True,
    },
    
    # =========================================================================
    # CONTROLLER - Especialista em CC, stun e slow
    # =========================================================================
    ('Controller', 'Warhammer'): {
        'name': 'Crowd Master',
        'description': 'Mestre do CC, stuna múltiplos inimigos',
        'aggression': 0.55,
        'preferred_distance': 60,
        'attack_distance': 65,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.9,  # Usa stun em área muito
        'ability_distance_threshold': 80,  # Usa quando inimigos agrupados
        'strafe_intensity': 0.4,
        'approach_speed': 0.5,
        'retreat_speed': 0.5,
        'combo_tendency': 0.45,
        'controller_mode': True,
        'cc_priority': True,
        'target_clustered': True,  # Prioriza grupos de inimigos
    },
    ('Controller', 'Staff'): {
        'name': 'Disruptor',
        'description': 'Controla de longe com magias de CC',
        'aggression': 0.35,
        'preferred_distance': 100,
        'attack_distance': 110,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.5,
        'approach_speed': 0.4,
        'retreat_speed': 0.65,
        'combo_tendency': 0.35,
        'controller_mode': True,
        'cc_priority': True,
    },
    ('Controller', 'Sword'): {
        'name': 'Battle Controller',
        'description': 'Controla no meio da luta',
        'aggression': 0.5,
        'preferred_distance': 55,
        'attack_distance': 70,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 70,
        'strafe_intensity': 0.45,
        'approach_speed': 0.55,
        'retreat_speed': 0.5,
        'combo_tendency': 0.5,
        'controller_mode': True,
        'cc_priority': True,
    },
    ('Controller', 'Tome'): {
        'name': 'Hex Mage',
        'description': 'Aplica debuffs e controla inimigos',
        'aggression': 0.3,
        'preferred_distance': 90,
        'attack_distance': 95,
        'retreat_health': 0.5,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.55,
        'approach_speed': 0.4,
        'retreat_speed': 0.6,
        'combo_tendency': 0.3,
        'controller_mode': True,
        'cc_priority': True,
        'debuff_priority': True,
    },
    
    # =========================================================================
    # RANGER - Atacante de longa distância
    # =========================================================================
    ('Ranger', 'Bow'): {
        'name': 'Sharpshooter',
        'description': 'Sniper puro, máximo dano à distância',
        'aggression': 0.5,
        'preferred_distance': 180,  # Máxima distância
        'attack_distance': 200,
        'retreat_health': 0.5,
        'ability_health_threshold': 0.7,  # Chuva de flechas
        'ability_distance_threshold': 150,
        'strafe_intensity': 0.6,
        'approach_speed': 0.3,
        'retreat_speed': 0.9,  # Recua muito rápido
        'combo_tendency': 0.7,
        'ranged_mode': True,
        'kite_enemies': True,  # Atira e recua
        'maintain_distance': True,
    },
    ('Ranger', 'Spear'): {
        'name': 'Javelin Thrower',
        'description': 'Combina arco com lança para alcance extremo',
        'aggression': 0.45,
        'preferred_distance': 150,
        'attack_distance': 120,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 140,
        'strafe_intensity': 0.5,
        'approach_speed': 0.4,
        'retreat_speed': 0.8,
        'combo_tendency': 0.5,
        'ranged_mode': True,
        'kite_enemies': True,
    },
    ('Ranger', 'Dagger'): {
        'name': 'Scout',
        'description': 'Ranger ágil, atira e escapa',
        'aggression': 0.55,
        'preferred_distance': 140,
        'attack_distance': 160,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 130,
        'strafe_intensity': 0.7,
        'approach_speed': 0.5,
        'retreat_speed': 0.95,
        'combo_tendency': 0.6,
        'ranged_mode': True,
        'kite_enemies': True,
        'hit_and_run': True,
    },
    ('Ranger', 'Sword'): {
        'name': 'Ranger Knight',
        'description': 'Ranger que pode lutar corpo a corpo se necessário',
        'aggression': 0.5,
        'preferred_distance': 120,
        'attack_distance': 140,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.5,
        'approach_speed': 0.55,
        'retreat_speed': 0.7,
        'combo_tendency': 0.55,
        'ranged_mode': True,
        'can_melee': True,
    },
    
    # =========================================================================
    # ENCHANTER - Buffer, fortalece aliados
    # =========================================================================
    ('Enchanter', 'Tome'): {
        'name': 'Grand Enchanter',
        'description': 'Mestre dos buffs, fortalece todo o time',
        'aggression': 0.25,
        'preferred_distance': 110,
        'attack_distance': 100,
        'retreat_health': 0.55,
        'ability_health_threshold': 0.95,  # Usa buff constantemente
        'ability_distance_threshold': 150,
        'strafe_intensity': 0.5,
        'approach_speed': 0.35,
        'retreat_speed': 0.7,
        'combo_tendency': 0.3,
        'enchanter_mode': True,
        'buff_priority': True,
        'stay_with_team': True,
    },
    ('Enchanter', 'Staff'): {
        'name': 'Mystic',
        'description': 'Combina buffs com cura leve',
        'aggression': 0.3,
        'preferred_distance': 100,
        'attack_distance': 95,
        'retreat_health': 0.5,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 140,
        'strafe_intensity': 0.45,
        'approach_speed': 0.4,
        'retreat_speed': 0.65,
        'combo_tendency': 0.35,
        'enchanter_mode': True,
        'buff_priority': True,
        'healer_mode': True,
    },
    ('Enchanter', 'Sword'): {
        'name': 'Battle Mage',
        'description': 'Encantador que luta na linha de frente',
        'aggression': 0.5,
        'preferred_distance': 60,
        'attack_distance': 70,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.4,
        'approach_speed': 0.55,
        'retreat_speed': 0.5,
        'combo_tendency': 0.55,
        'enchanter_mode': True,
        'buff_priority': True,
    },
    ('Enchanter', 'Warhammer'): {
        'name': 'Runeguard',
        'description': 'Buffs poderosos com capacidade de stun',
        'aggression': 0.45,
        'preferred_distance': 55,
        'attack_distance': 60,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.35,
        'approach_speed': 0.5,
        'retreat_speed': 0.45,
        'combo_tendency': 0.4,
        'enchanter_mode': True,
        'buff_priority': True,
        'stun_priority': True,
    },
    
    # =========================================================================
    # TRAPPER - Controle de terreno com armadilhas
    # =========================================================================
    ('Trapper', 'Trap_launcher'): {
        'name': 'Master Trapper',
        'description': 'Especialista em armadilhas, controla o terreno',
        'aggression': 0.4,
        'preferred_distance': 80,
        'attack_distance': 60,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.95,  # Coloca armadilhas constantemente
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.5,
        'approach_speed': 0.45,
        'retreat_speed': 0.6,
        'combo_tendency': 0.4,
        'trapper_mode': True,
        'zone_control': True,
        'trap_placement_priority': True,
    },
    ('Trapper', 'Bow'): {
        'name': 'Hunter',
        'description': 'Caçador que usa armadilhas e arco',
        'aggression': 0.45,
        'preferred_distance': 140,
        'attack_distance': 160,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.55,
        'approach_speed': 0.4,
        'retreat_speed': 0.75,
        'combo_tendency': 0.5,
        'trapper_mode': True,
        'zone_control': True,
        'ranged_mode': True,
    },
    ('Trapper', 'Dagger'): {
        'name': 'Saboteur',
        'description': 'Coloca armadilhas e ataca por trás',
        'aggression': 0.55,
        'preferred_distance': 70,
        'attack_distance': 50,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.65,
        'approach_speed': 0.6,
        'retreat_speed': 0.7,
        'combo_tendency': 0.7,
        'trapper_mode': True,
        'zone_control': True,
        'hit_and_run': True,
    },
    ('Trapper', 'Spear'): {
        'name': 'Warden',
        'description': 'Defende área com armadilhas e alcance',
        'aggression': 0.35,
        'preferred_distance': 100,
        'attack_distance': 110,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 110,
        'strafe_intensity': 0.4,
        'approach_speed': 0.4,
        'retreat_speed': 0.55,
        'combo_tendency': 0.4,
        'trapper_mode': True,
        'zone_control': True,
        'defensive_style': True,
    },
    
    # =========================================================================
    # ESTRATÉGIAS ADICIONAIS - TODAS AS COMBINAÇÕES RESTANTES
    # =========================================================================
    
    # WARRIOR com armas novas
    ('Warrior', 'Staff'): {
        'name': 'Mystic Warrior',
        'description': 'Guerreiro com poderes mágicos de suporte',
        'aggression': 0.5,
        'preferred_distance': 70,
        'attack_distance': 90,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.4,
        'approach_speed': 0.55,
        'retreat_speed': 0.5,
        'combo_tendency': 0.4,
    },
    ('Warrior', 'Bow'): {
        'name': 'Archer Soldier',
        'description': 'Guerreiro versátil que ataca de longe',
        'aggression': 0.5,
        'preferred_distance': 120,
        'attack_distance': 150,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.5,
        'approach_speed': 0.5,
        'retreat_speed': 0.6,
        'combo_tendency': 0.5,
        'ranged_mode': True,
    },
    ('Warrior', 'Warhammer'): {
        'name': 'Crushing Warrior',
        'description': 'Guerreiro com martelo pesado para stunnar',
        'aggression': 0.6,
        'preferred_distance': 50,
        'attack_distance': 60,
        'retreat_health': 0.25,
        'ability_health_threshold': 0.45,
        'ability_distance_threshold': 70,
        'strafe_intensity': 0.3,
        'approach_speed': 0.55,
        'retreat_speed': 0.45,
        'combo_tendency': 0.4,
        'stun_priority': True,
    },
    ('Warrior', 'Tome'): {
        'name': 'Scholar Warrior',
        'description': 'Guerreiro estudioso com magias de suporte',
        'aggression': 0.45,
        'preferred_distance': 80,
        'attack_distance': 90,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.45,
        'approach_speed': 0.5,
        'retreat_speed': 0.55,
        'combo_tendency': 0.35,
    },
    ('Warrior', 'Shield_bash'): {
        'name': 'Shield Champion',
        'description': 'Guerreiro focado em defesa com escudo',
        'aggression': 0.5,
        'preferred_distance': 45,
        'attack_distance': 55,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 60,
        'strafe_intensity': 0.25,
        'approach_speed': 0.5,
        'retreat_speed': 0.4,
        'combo_tendency': 0.45,
        'defensive_style': True,
    },
    ('Warrior', 'Trap_launcher'): {
        'name': 'Tactical Warrior',
        'description': 'Guerreiro que usa armadilhas taticamente',
        'aggression': 0.5,
        'preferred_distance': 70,
        'attack_distance': 60,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.6,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.4,
        'approach_speed': 0.5,
        'retreat_speed': 0.5,
        'combo_tendency': 0.45,
        'zone_control': True,
    },
    
    # BERSERKER com armas novas
    ('Berserker', 'Staff'): {
        'name': 'Rage Mage',
        'description': 'Berserker com poderes mágicos destrutivos',
        'aggression': 0.8,
        'preferred_distance': 60,
        'attack_distance': 80,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.6,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.3,
        'approach_speed': 0.75,
        'retreat_speed': 0.0,
        'combo_tendency': 0.5,
        'berserker_mode': True,
    },
    ('Berserker', 'Bow'): {
        'name': 'Savage Archer',
        'description': 'Arqueiro furioso que não recua',
        'aggression': 0.75,
        'preferred_distance': 100,
        'attack_distance': 140,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.4,
        'approach_speed': 0.7,
        'retreat_speed': 0.0,
        'combo_tendency': 0.6,
        'berserker_mode': True,
        'ranged_mode': True,
    },
    ('Berserker', 'Warhammer'): {
        'name': 'Rampaging Crusher',
        'description': 'Destruidor total com martelo',
        'aggression': 0.95,
        'preferred_distance': 45,
        'attack_distance': 60,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 70,
        'strafe_intensity': 0.2,
        'approach_speed': 0.8,
        'retreat_speed': 0.0,
        'combo_tendency': 0.4,
        'berserker_mode': True,
        'stun_priority': True,
    },
    ('Berserker', 'Tome'): {
        'name': 'Mad Scholar',
        'description': 'Estudioso enlouquecido pela raiva',
        'aggression': 0.7,
        'preferred_distance': 70,
        'attack_distance': 85,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.55,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.35,
        'approach_speed': 0.65,
        'retreat_speed': 0.0,
        'combo_tendency': 0.45,
        'berserker_mode': True,
    },
    ('Berserker', 'Shield_bash'): {
        'name': 'Charging Bull',
        'description': 'Berserker que avança com escudo',
        'aggression': 0.9,
        'preferred_distance': 40,
        'attack_distance': 55,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.5,
        'ability_distance_threshold': 60,
        'strafe_intensity': 0.2,
        'approach_speed': 0.85,
        'retreat_speed': 0.0,
        'combo_tendency': 0.5,
        'berserker_mode': True,
    },
    ('Berserker', 'Trap_launcher'): {
        'name': 'Wild Trapper',
        'description': 'Coloca armadilhas enquanto avança',
        'aggression': 0.75,
        'preferred_distance': 60,
        'attack_distance': 55,
        'retreat_health': 0.0,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.35,
        'approach_speed': 0.7,
        'retreat_speed': 0.0,
        'combo_tendency': 0.5,
        'berserker_mode': True,
        'zone_control': True,
    },
    
    # ASSASSIN com armas novas
    ('Assassin', 'Staff'): {
        'name': 'Shadow Mage',
        'description': 'Assassino com magia das sombras',
        'aggression': 0.55,
        'preferred_distance': 80,
        'attack_distance': 95,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 130,
        'strafe_intensity': 0.6,
        'approach_speed': 0.6,
        'retreat_speed': 0.8,
        'combo_tendency': 0.4,
        'hit_and_run': True,
        'stealth_approach': True,
    },
    ('Assassin', 'Bow'): {
        'name': 'Silent Hunter',
        'description': 'Caçador silencioso com flechas',
        'aggression': 0.5,
        'preferred_distance': 140,
        'attack_distance': 160,
        'retreat_health': 0.5,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 150,
        'strafe_intensity': 0.65,
        'approach_speed': 0.5,
        'retreat_speed': 0.9,
        'combo_tendency': 0.5,
        'hit_and_run': True,
        'stealth_approach': True,
        'ranged_mode': True,
    },
    ('Assassin', 'Warhammer'): {
        'name': 'Shadow Crusher',
        'description': 'Assassino com golpe de martelo surpresa',
        'aggression': 0.6,
        'preferred_distance': 55,
        'attack_distance': 60,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.5,
        'approach_speed': 0.65,
        'retreat_speed': 0.75,
        'combo_tendency': 0.3,
        'hit_and_run': True,
        'stealth_approach': True,
        'stun_priority': True,
    },
    ('Assassin', 'Tome'): {
        'name': 'Phantom Scholar',
        'description': 'Mestre das artes ocultas',
        'aggression': 0.45,
        'preferred_distance': 90,
        'attack_distance': 100,
        'retreat_health': 0.5,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.55,
        'approach_speed': 0.55,
        'retreat_speed': 0.8,
        'combo_tendency': 0.35,
        'hit_and_run': True,
        'stealth_approach': True,
    },
    ('Assassin', 'Shield_bash'): {
        'name': 'Shadow Guard',
        'description': 'Assassino defensivo que contra-ataca',
        'aggression': 0.5,
        'preferred_distance': 50,
        'attack_distance': 55,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.5,
        'approach_speed': 0.6,
        'retreat_speed': 0.7,
        'combo_tendency': 0.45,
        'hit_and_run': True,
        'stealth_approach': True,
        'defensive_style': True,
    },
    ('Assassin', 'Trap_launcher'): {
        'name': 'Ambush Master',
        'description': 'Mestre das emboscadas com armadilhas',
        'aggression': 0.5,
        'preferred_distance': 80,
        'attack_distance': 60,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.55,
        'approach_speed': 0.55,
        'retreat_speed': 0.75,
        'combo_tendency': 0.5,
        'hit_and_run': True,
        'stealth_approach': True,
        'zone_control': True,
    },
    
    # TANK com armas novas
    ('Tank', 'Staff'): {
        'name': 'Paladin',
        'description': 'Tank com poderes de cura',
        'aggression': 0.35,
        'preferred_distance': 60,
        'attack_distance': 80,
        'retreat_health': 0.15,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.25,
        'approach_speed': 0.4,
        'retreat_speed': 0.3,
        'combo_tendency': 0.35,
        'defensive_style': True,
        'healer_mode': True,
    },
    ('Tank', 'Bow'): {
        'name': 'Armored Archer',
        'description': 'Tank que ataca de longe',
        'aggression': 0.35,
        'preferred_distance': 100,
        'attack_distance': 130,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.65,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.3,
        'approach_speed': 0.35,
        'retreat_speed': 0.4,
        'combo_tendency': 0.4,
        'defensive_style': True,
        'ranged_mode': True,
    },
    ('Tank', 'Warhammer'): {
        'name': 'Earthshaker',
        'description': 'Tank devastador com martelo',
        'aggression': 0.45,
        'preferred_distance': 50,
        'attack_distance': 60,
        'retreat_health': 0.1,
        'ability_health_threshold': 0.6,
        'ability_distance_threshold': 65,
        'strafe_intensity': 0.2,
        'approach_speed': 0.4,
        'retreat_speed': 0.25,
        'combo_tendency': 0.35,
        'defensive_style': True,
        'stun_priority': True,
    },
    ('Tank', 'Tome'): {
        'name': 'Runic Guardian',
        'description': 'Tank com runas de proteção',
        'aggression': 0.3,
        'preferred_distance': 70,
        'attack_distance': 85,
        'retreat_health': 0.15,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.25,
        'approach_speed': 0.35,
        'retreat_speed': 0.3,
        'combo_tendency': 0.3,
        'defensive_style': True,
    },
    ('Tank', 'Shield_bash'): {
        'name': 'Wall of Steel',
        'description': 'Tank máximo com escudo duplo',
        'aggression': 0.3,
        'preferred_distance': 45,
        'attack_distance': 55,
        'retreat_health': 0.1,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 60,
        'strafe_intensity': 0.15,
        'approach_speed': 0.35,
        'retreat_speed': 0.2,
        'combo_tendency': 0.4,
        'defensive_style': True,
        'shield_reactive': True,
    },
    ('Tank', 'Trap_launcher'): {
        'name': 'Fortress Builder',
        'description': 'Tank que cria zona segura com armadilhas',
        'aggression': 0.3,
        'preferred_distance': 60,
        'attack_distance': 55,
        'retreat_health': 0.15,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.25,
        'approach_speed': 0.35,
        'retreat_speed': 0.3,
        'combo_tendency': 0.35,
        'defensive_style': True,
        'zone_control': True,
    },
    
    # LANCER com armas novas
    ('Lancer', 'Staff'): {
        'name': 'Mystic Rider',
        'description': 'Lancer com poderes mágicos',
        'aggression': 0.6,
        'preferred_distance': 85,
        'attack_distance': 90,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 110,
        'strafe_intensity': 0.45,
        'approach_speed': 0.6,
        'retreat_speed': 0.75,
        'combo_tendency': 0.4,
        'dash_in': True,
        'dash_out': True,
    },
    ('Lancer', 'Bow'): {
        'name': 'Swift Archer',
        'description': 'Arqueiro extremamente móvel',
        'aggression': 0.55,
        'preferred_distance': 130,
        'attack_distance': 150,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 140,
        'strafe_intensity': 0.6,
        'approach_speed': 0.55,
        'retreat_speed': 0.9,
        'combo_tendency': 0.55,
        'dash_in': True,
        'dash_out': True,
        'ranged_mode': True,
        'kite_enemies': True,
    },
    ('Lancer', 'Warhammer'): {
        'name': 'Charging Hammer',
        'description': 'Dash devastador com martelo',
        'aggression': 0.65,
        'preferred_distance': 75,
        'attack_distance': 65,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.35,
        'approach_speed': 0.6,
        'retreat_speed': 0.7,
        'combo_tendency': 0.35,
        'dash_in': True,
        'dash_out': True,
        'stun_priority': True,
    },
    ('Lancer', 'Tome'): {
        'name': 'Spell Knight',
        'description': 'Cavaleiro mágico veloz',
        'aggression': 0.55,
        'preferred_distance': 80,
        'attack_distance': 90,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.5,
        'approach_speed': 0.55,
        'retreat_speed': 0.7,
        'combo_tendency': 0.4,
        'dash_in': True,
        'dash_out': True,
    },
    ('Lancer', 'Shield_bash'): {
        'name': 'Shield Charger',
        'description': 'Avança com escudo e dash',
        'aggression': 0.6,
        'preferred_distance': 70,
        'attack_distance': 55,
        'retreat_health': 0.25,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.3,
        'approach_speed': 0.6,
        'retreat_speed': 0.65,
        'combo_tendency': 0.45,
        'dash_in': True,
        'dash_out': True,
        'defensive_style': True,
    },
    ('Lancer', 'Trap_launcher'): {
        'name': 'Hit and Trap',
        'description': 'Coloca armadilhas com mobilidade',
        'aggression': 0.55,
        'preferred_distance': 80,
        'attack_distance': 60,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.45,
        'approach_speed': 0.55,
        'retreat_speed': 0.7,
        'combo_tendency': 0.45,
        'dash_in': True,
        'dash_out': True,
        'zone_control': True,
    },
    
    # CLERIC com armas restantes
    ('Cleric', 'Greatsword'): {
        'name': 'Crusader',
        'description': 'Clérigo guerreiro com espada sagrada',
        'aggression': 0.5,
        'preferred_distance': 60,
        'attack_distance': 80,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.6,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.35,
        'approach_speed': 0.5,
        'retreat_speed': 0.5,
        'combo_tendency': 0.4,
        'healer_mode': True,
        'protect_allies': True,
    },
    ('Cleric', 'Dagger'): {
        'name': 'Shadow Priest',
        'description': 'Clérigo ágil com adagas',
        'aggression': 0.55,
        'preferred_distance': 45,
        'attack_distance': 50,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.65,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.55,
        'approach_speed': 0.65,
        'retreat_speed': 0.7,
        'combo_tendency': 0.65,
        'healer_mode': True,
        'protect_allies': True,
    },
    ('Cleric', 'Spear'): {
        'name': 'Holy Lancer',
        'description': 'Clérigo com lança sagrada',
        'aggression': 0.4,
        'preferred_distance': 90,
        'attack_distance': 110,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.65,
        'ability_distance_threshold': 130,
        'strafe_intensity': 0.45,
        'approach_speed': 0.45,
        'retreat_speed': 0.6,
        'combo_tendency': 0.4,
        'healer_mode': True,
        'protect_allies': True,
    },
    ('Cleric', 'Bow'): {
        'name': 'Divine Archer',
        'description': 'Arqueiro sagrado que cura',
        'aggression': 0.35,
        'preferred_distance': 130,
        'attack_distance': 150,
        'retreat_health': 0.5,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 140,
        'strafe_intensity': 0.5,
        'approach_speed': 0.4,
        'retreat_speed': 0.75,
        'combo_tendency': 0.5,
        'healer_mode': True,
        'protect_allies': True,
        'ranged_mode': True,
    },
    ('Cleric', 'Shield_bash'): {
        'name': 'Holy Guardian',
        'description': 'Clérigo protetor com escudo',
        'aggression': 0.35,
        'preferred_distance': 50,
        'attack_distance': 55,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.25,
        'approach_speed': 0.4,
        'retreat_speed': 0.4,
        'combo_tendency': 0.4,
        'healer_mode': True,
        'protect_allies': True,
        'defensive_style': True,
    },
    ('Cleric', 'Trap_launcher'): {
        'name': 'Sacred Trapper',
        'description': 'Clérigo que protege área com armadilhas',
        'aggression': 0.3,
        'preferred_distance': 80,
        'attack_distance': 60,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.4,
        'approach_speed': 0.4,
        'retreat_speed': 0.6,
        'combo_tendency': 0.35,
        'healer_mode': True,
        'protect_allies': True,
        'zone_control': True,
    },
    
    # GUARDIAN com armas restantes
    ('Guardian', 'Greatsword'): {
        'name': 'Great Protector',
        'description': 'Guardião com espada enorme',
        'aggression': 0.4,
        'preferred_distance': 55,
        'attack_distance': 80,
        'retreat_health': 0.15,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.2,
        'approach_speed': 0.4,
        'retreat_speed': 0.3,
        'combo_tendency': 0.35,
        'guardian_mode': True,
        'protect_allies': True,
    },
    ('Guardian', 'Dagger'): {
        'name': 'Swift Guardian',
        'description': 'Guardião ágil com adagas',
        'aggression': 0.45,
        'preferred_distance': 45,
        'attack_distance': 50,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.4,
        'approach_speed': 0.55,
        'retreat_speed': 0.4,
        'combo_tendency': 0.6,
        'guardian_mode': True,
        'protect_allies': True,
    },
    ('Guardian', 'Staff'): {
        'name': 'Warden Mage',
        'description': 'Guardião com poderes mágicos de proteção',
        'aggression': 0.3,
        'preferred_distance': 80,
        'attack_distance': 90,
        'retreat_health': 0.25,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.35,
        'approach_speed': 0.4,
        'retreat_speed': 0.4,
        'combo_tendency': 0.3,
        'guardian_mode': True,
        'protect_allies': True,
        'healer_mode': True,
    },
    ('Guardian', 'Bow'): {
        'name': 'Watchful Guardian',
        'description': 'Guardião que protege de longe',
        'aggression': 0.35,
        'preferred_distance': 110,
        'attack_distance': 130,
        'retreat_health': 0.25,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.4,
        'approach_speed': 0.4,
        'retreat_speed': 0.5,
        'combo_tendency': 0.45,
        'guardian_mode': True,
        'protect_allies': True,
        'ranged_mode': True,
    },
    ('Guardian', 'Tome'): {
        'name': 'Runic Warden',
        'description': 'Guardião com runas protetoras',
        'aggression': 0.3,
        'preferred_distance': 85,
        'attack_distance': 95,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 110,
        'strafe_intensity': 0.35,
        'approach_speed': 0.4,
        'retreat_speed': 0.45,
        'combo_tendency': 0.3,
        'guardian_mode': True,
        'protect_allies': True,
    },
    ('Guardian', 'Trap_launcher'): {
        'name': 'Defensive Trapper',
        'description': 'Guardião que cria zonas seguras',
        'aggression': 0.3,
        'preferred_distance': 70,
        'attack_distance': 55,
        'retreat_health': 0.2,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.3,
        'approach_speed': 0.4,
        'retreat_speed': 0.4,
        'combo_tendency': 0.35,
        'guardian_mode': True,
        'protect_allies': True,
        'zone_control': True,
    },
    
    # CONTROLLER com armas restantes
    ('Controller', 'Greatsword'): {
        'name': 'Heavy Controller',
        'description': 'Controlador com espada pesada',
        'aggression': 0.5,
        'preferred_distance': 60,
        'attack_distance': 80,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 85,
        'strafe_intensity': 0.35,
        'approach_speed': 0.5,
        'retreat_speed': 0.45,
        'combo_tendency': 0.35,
        'controller_mode': True,
        'cc_priority': True,
    },
    ('Controller', 'Dagger'): {
        'name': 'Swift Disabler',
        'description': 'Controlador rápido com adagas',
        'aggression': 0.55,
        'preferred_distance': 50,
        'attack_distance': 55,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.8,
        'ability_distance_threshold': 70,
        'strafe_intensity': 0.55,
        'approach_speed': 0.65,
        'retreat_speed': 0.6,
        'combo_tendency': 0.6,
        'controller_mode': True,
        'cc_priority': True,
    },
    ('Controller', 'Spear'): {
        'name': 'Long Range CC',
        'description': 'Controlador com alcance de lança',
        'aggression': 0.4,
        'preferred_distance': 95,
        'attack_distance': 110,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.45,
        'approach_speed': 0.45,
        'retreat_speed': 0.55,
        'combo_tendency': 0.4,
        'controller_mode': True,
        'cc_priority': True,
    },
    ('Controller', 'Bow'): {
        'name': 'Snaring Archer',
        'description': 'Arqueiro que controla de longe',
        'aggression': 0.4,
        'preferred_distance': 140,
        'attack_distance': 160,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 130,
        'strafe_intensity': 0.5,
        'approach_speed': 0.4,
        'retreat_speed': 0.7,
        'combo_tendency': 0.5,
        'controller_mode': True,
        'cc_priority': True,
        'ranged_mode': True,
    },
    ('Controller', 'Shield_bash'): {
        'name': 'Stunning Wall',
        'description': 'Controlador defensivo com escudo',
        'aggression': 0.45,
        'preferred_distance': 50,
        'attack_distance': 55,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 70,
        'strafe_intensity': 0.3,
        'approach_speed': 0.45,
        'retreat_speed': 0.4,
        'combo_tendency': 0.4,
        'controller_mode': True,
        'cc_priority': True,
        'defensive_style': True,
    },
    ('Controller', 'Trap_launcher'): {
        'name': 'Zone Controller',
        'description': 'Mestre do controle territorial',
        'aggression': 0.4,
        'preferred_distance': 80,
        'attack_distance': 60,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.95,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.45,
        'approach_speed': 0.45,
        'retreat_speed': 0.55,
        'combo_tendency': 0.4,
        'controller_mode': True,
        'cc_priority': True,
        'zone_control': True,
    },
    
    # RANGER com armas restantes
    ('Ranger', 'Greatsword'): {
        'name': 'Beast Slayer',
        'description': 'Ranger com espada para combate próximo',
        'aggression': 0.55,
        'preferred_distance': 70,
        'attack_distance': 85,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.4,
        'approach_speed': 0.55,
        'retreat_speed': 0.6,
        'combo_tendency': 0.4,
        'ranged_mode': True,
        'can_melee': True,
    },
    ('Ranger', 'Staff'): {
        'name': 'Nature Warden',
        'description': 'Ranger com magia da natureza',
        'aggression': 0.4,
        'preferred_distance': 110,
        'attack_distance': 100,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.5,
        'approach_speed': 0.4,
        'retreat_speed': 0.7,
        'combo_tendency': 0.4,
        'ranged_mode': True,
        'healer_mode': True,
    },
    ('Ranger', 'Warhammer'): {
        'name': 'Heavy Hunter',
        'description': 'Ranger com martelo para stunnar presas',
        'aggression': 0.5,
        'preferred_distance': 80,
        'attack_distance': 65,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.4,
        'approach_speed': 0.5,
        'retreat_speed': 0.6,
        'combo_tendency': 0.35,
        'ranged_mode': True,
        'stun_priority': True,
    },
    ('Ranger', 'Tome'): {
        'name': 'Arcane Hunter',
        'description': 'Ranger com conhecimento arcano',
        'aggression': 0.4,
        'preferred_distance': 120,
        'attack_distance': 110,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.75,
        'ability_distance_threshold': 130,
        'strafe_intensity': 0.5,
        'approach_speed': 0.4,
        'retreat_speed': 0.7,
        'combo_tendency': 0.4,
        'ranged_mode': True,
    },
    ('Ranger', 'Shield_bash'): {
        'name': 'Armored Hunter',
        'description': 'Ranger com defesa de escudo',
        'aggression': 0.45,
        'preferred_distance': 90,
        'attack_distance': 60,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.7,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.4,
        'approach_speed': 0.45,
        'retreat_speed': 0.6,
        'combo_tendency': 0.45,
        'ranged_mode': True,
        'defensive_style': True,
    },
    ('Ranger', 'Trap_launcher'): {
        'name': 'Master Hunter',
        'description': 'Caçador supremo com armadilhas',
        'aggression': 0.45,
        'preferred_distance': 100,
        'attack_distance': 60,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.5,
        'approach_speed': 0.45,
        'retreat_speed': 0.65,
        'combo_tendency': 0.45,
        'ranged_mode': True,
        'zone_control': True,
    },
    
    # ENCHANTER com armas restantes
    ('Enchanter', 'Greatsword'): {
        'name': 'Runic Blademaster',
        'description': 'Encantador com lâmina rúnica',
        'aggression': 0.5,
        'preferred_distance': 60,
        'attack_distance': 80,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.35,
        'approach_speed': 0.5,
        'retreat_speed': 0.5,
        'combo_tendency': 0.4,
        'enchanter_mode': True,
        'buff_priority': True,
    },
    ('Enchanter', 'Dagger'): {
        'name': 'Quick Enchanter',
        'description': 'Encantador ágil com adagas',
        'aggression': 0.5,
        'preferred_distance': 50,
        'attack_distance': 55,
        'retreat_health': 0.4,
        'ability_health_threshold': 0.85,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.55,
        'approach_speed': 0.6,
        'retreat_speed': 0.65,
        'combo_tendency': 0.65,
        'enchanter_mode': True,
        'buff_priority': True,
    },
    ('Enchanter', 'Spear'): {
        'name': 'Spell Lancer',
        'description': 'Encantador com lança mágica',
        'aggression': 0.4,
        'preferred_distance': 95,
        'attack_distance': 110,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 120,
        'strafe_intensity': 0.45,
        'approach_speed': 0.45,
        'retreat_speed': 0.6,
        'combo_tendency': 0.4,
        'enchanter_mode': True,
        'buff_priority': True,
    },
    ('Enchanter', 'Bow'): {
        'name': 'Arcane Archer',
        'description': 'Arqueiro com flechas encantadas',
        'aggression': 0.4,
        'preferred_distance': 140,
        'attack_distance': 160,
        'retreat_health': 0.5,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 150,
        'strafe_intensity': 0.5,
        'approach_speed': 0.4,
        'retreat_speed': 0.75,
        'combo_tendency': 0.55,
        'enchanter_mode': True,
        'buff_priority': True,
        'ranged_mode': True,
    },
    ('Enchanter', 'Shield_bash'): {
        'name': 'Ward Master',
        'description': 'Encantador protetor com escudo',
        'aggression': 0.35,
        'preferred_distance': 55,
        'attack_distance': 55,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.3,
        'approach_speed': 0.45,
        'retreat_speed': 0.45,
        'combo_tendency': 0.4,
        'enchanter_mode': True,
        'buff_priority': True,
        'defensive_style': True,
    },
    ('Enchanter', 'Trap_launcher'): {
        'name': 'Rune Trapper',
        'description': 'Encantador que coloca armadilhas mágicas',
        'aggression': 0.35,
        'preferred_distance': 85,
        'attack_distance': 60,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.45,
        'approach_speed': 0.45,
        'retreat_speed': 0.6,
        'combo_tendency': 0.4,
        'enchanter_mode': True,
        'buff_priority': True,
        'zone_control': True,
    },
    
    # TRAPPER com armas restantes
    ('Trapper', 'Sword'): {
        'name': 'Combat Trapper',
        'description': 'Trapper que também luta bem corpo a corpo',
        'aggression': 0.5,
        'preferred_distance': 60,
        'attack_distance': 70,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 90,
        'strafe_intensity': 0.45,
        'approach_speed': 0.55,
        'retreat_speed': 0.55,
        'combo_tendency': 0.55,
        'trapper_mode': True,
        'zone_control': True,
    },
    ('Trapper', 'Greatsword'): {
        'name': 'Heavy Trapper',
        'description': 'Trapper com espada pesada',
        'aggression': 0.45,
        'preferred_distance': 65,
        'attack_distance': 80,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 95,
        'strafe_intensity': 0.35,
        'approach_speed': 0.5,
        'retreat_speed': 0.5,
        'combo_tendency': 0.4,
        'trapper_mode': True,
        'zone_control': True,
    },
    ('Trapper', 'Staff'): {
        'name': 'Mystic Trapper',
        'description': 'Trapper com magia e armadilhas',
        'aggression': 0.35,
        'preferred_distance': 90,
        'attack_distance': 95,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.95,
        'ability_distance_threshold': 110,
        'strafe_intensity': 0.45,
        'approach_speed': 0.45,
        'retreat_speed': 0.6,
        'combo_tendency': 0.35,
        'trapper_mode': True,
        'zone_control': True,
    },
    ('Trapper', 'Warhammer'): {
        'name': 'Stunning Trapper',
        'description': 'Trapper que stuna e coloca armadilhas',
        'aggression': 0.45,
        'preferred_distance': 60,
        'attack_distance': 60,
        'retreat_health': 0.35,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 85,
        'strafe_intensity': 0.4,
        'approach_speed': 0.5,
        'retreat_speed': 0.5,
        'combo_tendency': 0.4,
        'trapper_mode': True,
        'zone_control': True,
        'stun_priority': True,
    },
    ('Trapper', 'Tome'): {
        'name': 'Scholar Trapper',
        'description': 'Trapper estudioso com conhecimento arcano',
        'aggression': 0.35,
        'preferred_distance': 85,
        'attack_distance': 90,
        'retreat_health': 0.45,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 100,
        'strafe_intensity': 0.45,
        'approach_speed': 0.45,
        'retreat_speed': 0.55,
        'combo_tendency': 0.35,
        'trapper_mode': True,
        'zone_control': True,
    },
    ('Trapper', 'Shield_bash'): {
        'name': 'Defensive Trapper',
        'description': 'Trapper defensivo com escudo',
        'aggression': 0.35,
        'preferred_distance': 55,
        'attack_distance': 55,
        'retreat_health': 0.3,
        'ability_health_threshold': 0.9,
        'ability_distance_threshold': 80,
        'strafe_intensity': 0.3,
        'approach_speed': 0.45,
        'retreat_speed': 0.45,
        'combo_tendency': 0.4,
        'trapper_mode': True,
        'zone_control': True,
        'defensive_style': True,
    },
}


class StrategicAI(AIController):
    """
    IA com estratégias específicas baseadas na combinação classe + arma.
    Cada uma das 20 combinações tem comportamento único otimizado.
    """
    
    def __init__(self, class_name: str, weapon_name: str):
        super().__init__()
        self.class_name = class_name
        self.weapon_name = weapon_name
        
        # Carrega a estratégia específica
        key = (class_name, weapon_name)
        if key in CLASS_WEAPON_STRATEGIES:
            self.strategy = CLASS_WEAPON_STRATEGIES[key]
        else:
            # Fallback para estratégia genérica
            self.strategy = {
                'name': 'Generic',
                'aggression': 0.6,
                'preferred_distance': 60,
                'attack_distance': 75,
                'retreat_health': 0.25,
                'ability_health_threshold': 0.4,
                'ability_distance_threshold': 80,
                'strafe_intensity': 0.4,
                'approach_speed': 0.7,
                'retreat_speed': 0.5,
                'combo_tendency': 0.5,
            }
        
        # Estado interno para comportamentos complexos
        self.attack_cooldown_frames = 0
        self.last_attack_time = 0
        self.in_retreat = False
        self.stealth_active = False
    
    def decide_actions(self, observation: Dict) -> Dict:
        actions = {
            'move_x': 0,
            'move_y': 0,
            'attack': False,
            'ability': False
        }
        
        if not observation.get('enemies'):
            # Sem inimigos visíveis - movimento aleatório leve
            if self.strategy.get('stealth_approach'):
                # Assassino pode usar invisibilidade para procurar
                self_state = observation['self']
                if self_state['ability_ready']:
                    actions['ability'] = True
            return actions
        
        self_state = observation['self']
        strat = self.strategy
        
        # Encontrar inimigo mais próximo
        closest_enemy = None
        min_distance = float('inf')
        
        for enemy in observation['enemies']:
            dx = enemy['x'] - self_state['x']
            dy = enemy['y'] - self_state['y']
            dist = math.sqrt(dx*dx + dy*dy)
            if dist < min_distance:
                min_distance = dist
                closest_enemy = enemy
        
        if not closest_enemy:
            return actions
        
        # Calcular direção para o inimigo
        dx = closest_enemy['x'] - self_state['x']
        dy = closest_enemy['y'] - self_state['y']
        distance = min_distance
        
        if distance > 0:
            dx /= distance
            dy /= distance
        
        health_ratio = self_state['health_ratio']
        weapon_state = self_state.get('weapon', {})
        can_attack = weapon_state.get('can_attack', False)
        ability_ready = self_state['ability_ready']
        
        # =================================================================
        # LÓGICA DE HABILIDADE ESPECIAL
        # =================================================================
        should_use_ability = False
        
        # Berserker - usa rage para buff de dano (especialmente com vida baixa)
        if strat.get('berserker_mode'):
            if ability_ready:
                if health_ratio < strat['ability_health_threshold']:
                    should_use_ability = True
                elif distance < strat['ability_distance_threshold']:
                    should_use_ability = True
        
        # Assassin - usa invisibilidade para aproximar ou escapar
        elif strat.get('stealth_approach'):
            if ability_ready:
                if distance > strat['ability_distance_threshold']:
                    should_use_ability = True
                elif health_ratio < strat['ability_health_threshold'] * 0.5:
                    should_use_ability = True
        
        # Tank - usa escudo reativamente
        elif strat.get('shield_reactive'):
            if ability_ready:
                if health_ratio < strat['ability_health_threshold']:
                    should_use_ability = True
                elif distance < strat['ability_distance_threshold']:
                    should_use_ability = True
        
        # Lancer - usa dash para entrar ou sair
        elif strat.get('dash_in') or strat.get('dash_out'):
            if ability_ready:
                if strat.get('dash_in') and distance > strat['ability_distance_threshold']:
                    should_use_ability = True
                elif strat.get('dash_out') and health_ratio < 0.3:
                    should_use_ability = True
        
        # NOVAS CLASSES DE SUPORTE/CONTROLE
        
        # Healer - usa cura quando aliados ou si mesmo estão com pouca vida
        elif strat.get('healer_mode'):
            if ability_ready:
                # Verifica se está com vida baixa
                if health_ratio < strat['ability_health_threshold']:
                    should_use_ability = True
                # Em combate em grupo, verificaria aliados também
        
        # Guardian - usa escudo protetor quando inimigo aproxima
        elif strat.get('guardian_mode'):
            if ability_ready:
                if distance < strat['ability_distance_threshold']:
                    should_use_ability = True
                elif health_ratio < strat['ability_health_threshold']:
                    should_use_ability = True
        
        # Controller - usa CC quando inimigo está no alcance
        elif strat.get('controller_mode'):
            if ability_ready:
                # Usa stun/cc quando inimigo está perto
                if distance < strat['ability_distance_threshold']:
                    should_use_ability = True
        
        # Ranger - usa chuva de flechas em área
        elif strat.get('ranged_mode'):
            if ability_ready:
                # Usa quando inimigo está no alcance da habilidade
                if distance < strat['ability_distance_threshold'] and distance > 60:
                    should_use_ability = True
        
        # Enchanter - usa buffs constantemente
        elif strat.get('enchanter_mode'):
            if ability_ready:
                # Sempre usa buff quando disponível
                should_use_ability = True
        
        # Trapper - coloca armadilhas estrategicamente
        elif strat.get('trapper_mode'):
            if ability_ready:
                # Coloca armadilha quando inimigo está se aproximando
                if distance < strat['ability_distance_threshold'] and distance > 40:
                    should_use_ability = True
        
        # Warrior - usa escudo defensivamente (fallback)
        else:
            if ability_ready:
                if health_ratio < strat['ability_health_threshold']:
                    should_use_ability = True
                elif distance < strat['ability_distance_threshold']:
                    should_use_ability = True
        
        actions['ability'] = should_use_ability
        
        # =================================================================
        # LÓGICA DE MOVIMENTO
        # =================================================================
        
        # Verificar se deve recuar (exceto berserker)
        should_retreat = False
        if not strat.get('berserker_mode'):
            if health_ratio < strat['retreat_health']:
                should_retreat = True
                self.in_retreat = True
        
        # Hit and run para assassinos e rangers
        if strat.get('hit_and_run') or strat.get('kite_enemies'):
            if self.attack_cooldown_frames > 0:
                self.attack_cooldown_frames -= 1
                should_retreat = True
        
        # Estilo defensivo - mantém distância
        if strat.get('defensive_style'):
            if distance < strat['preferred_distance'] - 30:
                should_retreat = True
        
        # Ranged sempre tenta manter distância máxima
        if strat.get('maintain_distance') or strat.get('kite_enemies'):
            if distance < strat['preferred_distance'] - 40:
                should_retreat = True
        
        if should_retreat:
            # Recuar do inimigo
            retreat_speed = strat['retreat_speed']
            actions['move_x'] = -dx * retreat_speed
            actions['move_y'] = -dy * retreat_speed
            
            # Adiciona movimento lateral para dificultar perseguição
            actions['move_x'] += -dy * 0.3
            actions['move_y'] += dx * 0.3
            
        elif distance > strat['preferred_distance'] + 20:
            # Aproximar do inimigo
            approach_speed = strat['approach_speed']
            
            # Berserker fica mais rápido com vida baixa
            if strat.get('berserker_mode') and health_ratio < 0.5:
                approach_speed = min(1.0, approach_speed * 1.3)
            
            # Ranged se aproxima devagar e com cuidado
            if strat.get('ranged_mode'):
                approach_speed *= 0.5
            
            actions['move_x'] = dx * approach_speed
            actions['move_y'] = dy * approach_speed
            
        elif distance < strat['preferred_distance'] - 20:
            # Muito perto - recuar levemente (se permitido)
            if strat['retreat_speed'] > 0:
                actions['move_x'] = -dx * strat['retreat_speed'] * 0.3
                actions['move_y'] = -dy * strat['retreat_speed'] * 0.3
            else:
                # Berserker não recua, faz strafe
                actions['move_x'] = -dy * strat['strafe_intensity']
                actions['move_y'] = dx * strat['strafe_intensity']
        else:
            # Na distância ideal - strafe
            strafe = strat['strafe_intensity']
            actions['move_x'] = -dy * strafe
            actions['move_y'] = dx * strafe
        
        # =================================================================
        # LÓGICA DE ATAQUE
        # =================================================================
        
        if can_attack and distance < strat['attack_distance']:
            # Verificar condições especiais
            should_attack = True
            
            # Estilo defensivo - só ataca em contra-ataque
            if strat.get('defensive_style'):
                if distance > strat['preferred_distance'] + 10:
                    should_attack = False
            
            # Healers preferem não atacar (focam em cura)
            if strat.get('healer_mode') and not strat.get('buff_priority'):
                if health_ratio > 0.5:  # Se não está precisando atacar para sobreviver
                    should_attack = distance < 50  # Só ataca se muito perto
            
            if should_attack:
                actions['attack'] = True
                
                # Hit and run / kite - marca que atacou
                if strat.get('hit_and_run') or strat.get('kite_enemies'):
                    self.attack_cooldown_frames = 30
        
        return actions
    
    def get_strategy_info(self) -> Dict:
        """Retorna informações sobre a estratégia atual."""
        return {
            'class': self.class_name,
            'weapon': self.weapon_name,
            'strategy_name': self.strategy.get('name', 'Unknown'),
            'description': self.strategy.get('description', ''),
            'aggression': self.strategy.get('aggression', 0.5),
            'preferred_distance': self.strategy.get('preferred_distance', 60),
        }


class NeuralNetworkAI(AIController):
    """
    Controlador para IA baseada em rede neural.
    Compatível com PyTorch, TensorFlow, ou qualquer framework.
    """
    
    def __init__(self, model=None):
        super().__init__()
        self.model = model  # O modelo de rede neural
    
    def set_model(self, model):
        """Define o modelo de rede neural"""
        self.model = model
    
    def decide_actions(self, observation: Dict) -> Dict:
        """
        Usa a rede neural para decidir ações.
        """
        if self.model is None:
            # Fallback para IA simples se não houver modelo
            return SimpleAI().decide_actions(observation)
        
        # Converter observação para vetor
        obs_vector = self.get_observation_vector()
        
        # Fazer inferência
        # O formato exato depende do framework usado
        try:
            # Para PyTorch
            import torch
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs_vector).unsqueeze(0)
                action_vector = self.model(obs_tensor).squeeze().numpy()
        except ImportError:
            try:
                # Para TensorFlow/Keras
                import tensorflow as tf
                obs_tensor = tf.expand_dims(obs_vector, 0)
                action_vector = self.model(obs_tensor).numpy().squeeze()
            except ImportError:
                # Fallback: usa o modelo como função callable
                action_vector = self.model(obs_vector)
        
        return self.actions_from_vector(action_vector)


# ============================================================================
# UTILITÁRIOS PARA TREINAMENTO
# ============================================================================

class RewardCalculator:
    """Calcula recompensas para treinamento por reforço"""
    
    @staticmethod
    def calculate_reward(entity: 'Entity', enemies: List['Entity'], 
                         prev_state: Dict, curr_state: Dict) -> float:
        """
        Calcula a recompensa para um passo de treinamento.
        
        Args:
            entity: A entidade sendo treinada
            enemies: Lista de inimigos
            prev_state: Estado anterior
            curr_state: Estado atual
        
        Returns:
            float: Recompensa
        """
        reward = 0.0
        
        # Recompensa por dano causado
        for enemy in enemies:
            if hasattr(enemy, 'last_damage_source') and enemy.last_damage_source == entity:
                reward += 1.0
        
        # Penalidade por dano recebido
        health_diff = curr_state['health'] - prev_state['health']
        if health_diff < 0:
            reward += health_diff * 0.1  # Penalidade proporcional
        
        # Recompensa por matar inimigo
        for enemy in enemies:
            if not enemy.is_alive():
                reward += 10.0
        
        # Penalidade por morrer
        if not entity.is_alive():
            reward -= 10.0
        
        # Pequena recompensa por se manter vivo
        reward += 0.01
        
        return reward
