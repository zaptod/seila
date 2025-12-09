"""
Circle Warriors - Weapon Ball Fight
====================================
Jogo 2D de combate com círculos armados.
Suporte para jogadores humanos e IA (redes neurais).
"""

import pygame
import math
import sys
import os
import random
import threading
import time
import json
import numpy as np
from typing import List, Optional, Dict
from collections import deque

# Imports do jogo
from entities import Entity, ClassRegistry, Warrior, Berserker, Assassin, Tank, Lancer
from weapons import WeaponRegistry
from physics import Physics
from controller import PlayerController, SimpleAI, StrategicAI, AIController
from game_state import GameState, ArenaConfig, GameConfig
from maps import MapRegistry, MapRenderer
from config_db import get_config_db, ConfigDatabase
from fog_of_war import FogOfWar, ObstacleManager, Camera, get_vision_radius

# Tentar importar PyTorch para redes neurais
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    print("✅ PyTorch disponível - Treinamento com Rede Neural ATIVO")
except ImportError:
    print("⚠️ PyTorch não encontrado - Instale com: pip install torch")
    print("   Treinamento usará IA simples como fallback")

# Inicialização do Pygame
pygame.init()

# Obter informações do monitor para adaptar a janela
display_info = pygame.display.Info()
MONITOR_WIDTH = display_info.current_w
MONITOR_HEIGHT = display_info.current_h

# Configurações da tela (90% do tamanho do monitor, respeitando proporção)
SCREEN_WIDTH = int(MONITOR_WIDTH * 0.9)
SCREEN_HEIGHT = int(MONITOR_HEIGHT * 0.85)

# Garantir tamanho mínimo
SCREEN_WIDTH = max(SCREEN_WIDTH, 1024)
SCREEN_HEIGHT = max(SCREEN_HEIGHT, 700)

# Criar janela centralizada e redimensionável
os.environ['SDL_VIDEO_CENTERED'] = '1'
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Circle Warriors - Weapon Ball Fight")

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
DARK_GRAY = (30, 30, 30)

# Clock para controle de FPS
clock = pygame.time.Clock()
FPS = 60

# Fontes escaláveis baseadas no tamanho da tela
def get_font_sizes():
    base = min(SCREEN_WIDTH, SCREEN_HEIGHT)
    return {
        'small': max(18, int(base * 0.022)),
        'medium': max(28, int(base * 0.035)),
        'large': max(40, int(base * 0.05))
    }

font_sizes = get_font_sizes()
font = pygame.font.Font(None, font_sizes['small'])
font_large = pygame.font.Font(None, font_sizes['medium'])
font_title = pygame.font.Font(None, font_sizes['large'])


# ============================================================================
# REDE NEURAL PARA TREINAMENTO
# ============================================================================

if TORCH_AVAILABLE:
    class ActorCritic(nn.Module):
        """Rede neural Actor-Critic para PPO"""
        
        def __init__(self, obs_size: int = 19, action_size: int = 4, hidden_size: int = 256):
            super().__init__()
            
            # Camadas compartilhadas
            self.shared = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            
            # Actor (política) - saída para ações
            self.actor_mean = nn.Linear(hidden_size, action_size)
            self.actor_log_std = nn.Parameter(torch.zeros(action_size))
            
            # Critic (valor)
            self.critic = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            shared = self.shared(x)
            action_mean = torch.tanh(self.actor_mean(shared))
            action_std = torch.exp(self.actor_log_std)
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


class Game:
    """Classe principal do jogo"""
    
    def __init__(self):
        self.arena_config = ArenaConfig(SCREEN_WIDTH, SCREEN_HEIGHT, 50)
        self.game_config = GameConfig()
        self.game_state = GameState(self.arena_config, self.game_config)
        
        self.physics = Physics(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.entities: List[Entity] = []
        
        self.running = True
        self.paused = False
        self.game_over = False
        self.winner: Optional[Entity] = None
        
        # Sistema de slow motion (câmera lenta)
        self.slow_motion = False
        self.slow_motion_timer = 0
        self.slow_motion_duration = 0.5  # Duração em segundos reais
        self.slow_motion_scale = 0.1  # 10% da velocidade normal (mais lento)
        self.last_entity_health = {}  # Para detectar dano
        
        # Modo de jogo
        self.mode = "menu"  # "menu", "select_p1", "select_p2", "pvp", "pve", "ai_vs_ai"
        
        # Seleção de classe e arma
        self.available_classes = ClassRegistry.list_classes()
        self.available_weapons = WeaponRegistry.list_weapons()
        
        self.selected_class_p1 = 0  # Índice
        self.selected_weapon_p1 = 0
        self.selected_class_p2 = 0
        self.selected_weapon_p2 = 0
        
        self.current_player_selecting = 1  # 1 ou 2
        self.selection_step = "class"  # "class", "weapon", "map" ou "ai_model"
        
        self.ai_difficulty = 0.7
        self.game_mode_choice = "pvp"  # Guarda a escolha para depois da seleção
        
        # Sistema de mapas
        self.available_maps = MapRegistry.list_maps()
        self.selected_map = 0  # Índice do mapa selecionado
        self.current_map = "arena"  # Mapa atual
        self.map_renderer = MapRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.map_time = 0  # Para animações dos mapas
        
        # Sistema de mapas grandes com Fog of War
        self.large_map_enabled = False  # Se True, usa mapa grande com fog of war
        self.large_map_width = 3000  # Largura do mapa grande
        self.large_map_height = 3000  # Altura do mapa grande
        self.fog_of_war = None  # Sistema de fog of war
        self.obstacle_manager = None  # Gerenciador de obstáculos
        self.camera = None  # Câmera para seguir jogador
        self.camera_target = None  # Entidade que a câmera segue
        
        # Sistema de IA treinada para modos normais
        self.use_trained_ai = False  # Se True, usa IA treinada ao invés de SimpleAI
        self.selected_ai_model_idx = 0  # Índice do modelo de IA selecionado
        self.game_ai_models = []  # Lista de modelos disponíveis para gameplay
        self.loaded_game_ai = None  # Rede neural carregada para gameplay
        
        # Sistema de treinamento de IA
        self.training_mode = False
        self.training_class = 0  # Índice da classe sendo treinada
        self.training_weapon = 0  # Índice da arma sendo treinada
        self.training_opponent_class = 0
        self.training_opponent_weapon = 0
        self.training_selection_step = "agent_class"  # agent_class, agent_weapon, opponent_class, opponent_weapon
        
        # Treinamento contra múltiplos oponentes
        self.training_multi_opponent = False  # Se True, treina contra vários tipos de oponentes
        self.training_opponent_pool = []  # Lista de (classe, arma) para treinar contra
        self.current_opponent_idx = 0  # Índice do oponente atual no pool
        self.episodes_per_opponent = 50  # Episódios antes de trocar de oponente
        
        # Configurações de treinamento
        self.training_episodes = 10000
        self.training_speed = 1  # 1 = normal, 2 = 2x, 0 = máximo
        self.training_render = True
        self.training_save_freq = 1000
        
        # Estado do treinamento
        self.trainer = None
        self.training_running = False
        self.training_paused = False
        self.training_stats = {
            'episode': 0,
            'total_episodes': 0,
            'wins': 0,
            'losses': 0,
            'avg_reward': 0,
            'recent_rewards': deque(maxlen=100),
            'best_reward': float('-inf'),
            'training_time': 0
        }
        self.training_entity = None
        self.training_opponent = None
        
        # Rede Neural para treinamento
        self.neural_network = None
        self.nn_optimizer = None
        self.nn_obs_size = 19  # Tamanho da observação
        self.nn_action_size = 4  # move_x, move_y, attack, ability
        
        # Buffer de experiência para PPO
        self.experience_buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Hiperparâmetros PPO
        self.ppo_gamma = 0.99
        self.ppo_gae_lambda = 0.95
        self.ppo_clip_epsilon = 0.2
        self.ppo_epochs = 10
        self.ppo_batch_size = 64
        self.ppo_update_freq = 2048  # Passos antes de atualizar
        self.training_step_count = 0
        
        # Feedback de salvamento
        self.save_message = ""
        self.save_message_timer = 0
        
        # Sistema de carregar modelos
        self.available_models = []
        self.selected_model_idx = 0
        self.load_menu_active = False
        
        # Banco de dados centralizado de configurações
        self.config_db = get_config_db()
        self.config_db.initialize_defaults(self.available_classes, self.available_weapons)
        
        # Editor de atributos (usa o banco de dados centralizado)
        self.editor_mode = None  # "classes" ou "weapons"
        self.editor_selected_item = 0
        self.editor_selected_stat = 0
        
        # Visualização de estatísticas de modelos
        self.model_stats_view_active = False
        self.model_stats_selected = 0
        
        # ============================================
        # SISTEMA DE TREINAMENTO EM GRUPO INTEGRADO
        # ============================================
        self.team_training_mode = False  # Se está em modo de treino em grupo
        self.team_size = 2  # 2v2, 3v3, ou 5v5
        self.team_battle_active = False  # Se uma batalha em grupo está ativa
        
        # Configuração das equipes
        self.blue_team = []  # Lista de entidades do time azul
        self.red_team = []   # Lista de entidades do time vermelho
        self.blue_team_config = []  # Config: [{"class": x, "weapon": y, "role": z}, ...]
        self.red_team_config = []
        
        # Papéis disponíveis
        self.available_roles = ["dps_melee", "dps_ranged", "tank", "healer", "controller", "support"]
        
        # Mapeamento de papel para combinações recomendadas
        self.role_recommendations = {
            "dps_melee": [("warrior", "sword"), ("berserker", "greatsword"), ("assassin", "dagger"), ("lancer", "spear")],
            "dps_ranged": [("ranger", "bow")],
            "tank": [("guardian", "shield_bash"), ("tank", "spear"), ("tank", "sword")],
            "healer": [("cleric", "staff")],
            "controller": [("controller", "warhammer"), ("trapper", "trap_launcher")],
            "support": [("enchanter", "tome")],
        }
        
        # Composições pré-definidas
        self.team_compositions = {
            "2v2_balanced": [
                {"class": "warrior", "weapon": "sword", "role": "dps_melee"},
                {"class": "cleric", "weapon": "staff", "role": "healer"},
            ],
            "2v2_aggressive": [
                {"class": "berserker", "weapon": "greatsword", "role": "dps_melee"},
                {"class": "assassin", "weapon": "dagger", "role": "dps_melee"},
            ],
            "2v2_defensive": [
                {"class": "guardian", "weapon": "shield_bash", "role": "tank"},
                {"class": "cleric", "weapon": "staff", "role": "healer"},
            ],
            "3v3_meta": [
                {"class": "guardian", "weapon": "shield_bash", "role": "tank"},
                {"class": "warrior", "weapon": "sword", "role": "dps_melee"},
                {"class": "cleric", "weapon": "staff", "role": "healer"},
            ],
            "5v5_meta": [
                {"class": "guardian", "weapon": "shield_bash", "role": "tank"},
                {"class": "berserker", "weapon": "greatsword", "role": "dps_melee"},
                {"class": "ranger", "weapon": "bow", "role": "dps_ranged"},
                {"class": "cleric", "weapon": "staff", "role": "healer"},
                {"class": "controller", "weapon": "warhammer", "role": "controller"},
            ],
        }
        
        # Seleção no menu de grupo
        self.group_menu_state = "main"  # main, select_size, select_blue, select_red, battle
        self.group_selected_composition = 0
        self.group_selecting_team = "blue"
        
        # Estatísticas de batalha em grupo
        self.group_battle_stats = {
            "blue_wins": 0,
            "red_wins": 0,
            "draws": 0,
            "total_battles": 0,
        }
        
        # Treinamento de rede neural em grupo
        self.group_training_active = False
        self.group_training_agent_idx = 0  # Qual membro do time está sendo treinado
        self.group_nn_obs_size = 50  # Observação maior para contexto de grupo
        self.group_neural_network = None
        self.group_nn_optimizer = None
        
        # Recompensas por papel
        self.role_reward_weights = {
            "dps_melee": {"damage_dealt": 0.15, "kills": 2.0, "damage_taken": -0.02, "death": -1.0},
            "dps_ranged": {"damage_dealt": 0.15, "kills": 2.0, "damage_taken": -0.03, "death": -1.5},
            "tank": {"damage_taken": 0.03, "allies_protected": 0.1, "death": -2.0, "cc_applied": 0.1},
            "healer": {"healing_done": 0.3, "allies_alive": 0.05, "death": -3.0, "damage_taken": -0.01},
            "controller": {"cc_applied": 0.25, "enemies_controlled": 0.1, "death": -1.5},
            "support": {"buffs_applied": 0.3, "allies_buffed": 0.1, "death": -1.5},
        }
    
    def create_entity(self, class_id: str, x: float, y: float, 
                      color: tuple, controller=None, weapon_id: str = None) -> Entity:
        """Cria uma nova entidade usando configurações do banco de dados central"""
        entity = ClassRegistry.create(class_id, x, y, color)
        
        # Aplicar stats customizados do banco de dados central
        self._apply_custom_stats(entity, class_id)
        
        # Definir arma customizada se especificada
        if weapon_id:
            entity.set_weapon(weapon_id)
            # Aplicar stats customizados da arma
            self._apply_custom_weapon_stats(entity, weapon_id)
        
        if controller:
            entity.set_controller(controller)
        
        # Dar acesso à lista global de entidades (para habilidades que precisam)
        entity.game_entities = self.entities
        
        self.entities.append(entity)
        self.game_state.add_entity(entity)
        entity._was_alive = True
        
        return entity
    
    def _apply_custom_stats(self, entity: Entity, class_id: str):
        """Aplica os stats customizados do banco de dados central à entidade"""
        custom_stats = self.config_db.get_class_stats(class_id)
        if custom_stats:
            stats = entity.stats_manager.get_stats()
            
            if 'max_health' in custom_stats:
                stats.max_health = custom_stats['max_health']
                entity.max_health = custom_stats['max_health']
                entity.health = custom_stats['max_health']
            if 'speed' in custom_stats:
                stats.speed = custom_stats['speed']
            if 'acceleration' in custom_stats:
                stats.acceleration = custom_stats['acceleration']
            if 'defense' in custom_stats:
                stats.defense = custom_stats['defense']
            if 'damage_multiplier' in custom_stats:
                stats.damage_multiplier = custom_stats['damage_multiplier']
    
    def _apply_custom_weapon_stats(self, entity: Entity, weapon_id: str):
        """Aplica os stats customizados da arma do banco de dados central"""
        custom_stats = self.config_db.get_weapon_stats(weapon_id)
        if custom_stats and entity.weapon:
            weapon_stats = entity.weapon.stats
            
            if 'base_damage' in custom_stats:
                weapon_stats.base_damage = custom_stats['base_damage']
            if 'range' in custom_stats:
                weapon_stats.range = custom_stats['range']
            if 'attack_cooldown' in custom_stats:
                weapon_stats.attack_cooldown = custom_stats['attack_cooldown']
            if 'knockback_force' in custom_stats:
                weapon_stats.knockback_force = custom_stats['knockback_force']
            if 'critical_chance' in custom_stats:
                weapon_stats.critical_chance = custom_stats['critical_chance']
    
    def get_selected_class_id(self, player: int) -> str:
        """Retorna o ID da classe selecionada"""
        idx = self.selected_class_p1 if player == 1 else self.selected_class_p2
        return self.available_classes[idx]
    
    def get_selected_weapon_id(self, player: int) -> str:
        """Retorna o ID da arma selecionada"""
        idx = self.selected_weapon_p1 if player == 1 else self.selected_weapon_p2
        return self.available_weapons[idx]
    
    def _create_neural_ai_controller(self):
        """Cria um controlador de IA usando a rede neural carregada"""
        game_ref = self  # Referência ao jogo para usar SCREEN_WIDTH/HEIGHT
        
        class NeuralGameAI(AIController):
            def __init__(self, neural_network, game):
                super().__init__()
                self.neural_network = neural_network
                self.game = game
            
            def _get_observation_19(self):
                """Observação de 19 valores compatível com o treinamento"""
                if not self.entity or not self.enemies:
                    return np.zeros(19, dtype=np.float32)
                
                agent = self.entity
                opponent = self.enemies[0] if self.enemies else None
                
                if not opponent or not opponent.is_alive():
                    return np.zeros(19, dtype=np.float32)
                
                agent_stats = agent.stats_manager.get_stats()
                opponent_stats = opponent.stats_manager.get_stats()
                
                obs = []
                
                # Estado do agente (9 valores)
                obs.extend([
                    agent.x / SCREEN_WIDTH,
                    agent.y / SCREEN_HEIGHT,
                    agent.vx / 20,
                    agent.vy / 20,
                    agent.facing_angle / math.pi,
                    agent.health / agent_stats.max_health,
                    1.0 if agent.invulnerable_time > 0 else 0.0,
                    1.0 if agent.weapon and agent.weapon.can_attack else 0.0,
                    1.0 if agent.ability_cooldown <= 0 else 0.0
                ])
                
                # Estado do oponente relativo (8 valores)
                rel_x = (opponent.x - agent.x) / SCREEN_WIDTH
                rel_y = (opponent.y - agent.y) / SCREEN_HEIGHT
                distance = math.sqrt(rel_x**2 + rel_y**2)
                angle_to = math.atan2(rel_y, rel_x)
                
                obs.extend([
                    rel_x,
                    rel_y,
                    distance,
                    angle_to / math.pi,
                    opponent.vx / 20,
                    opponent.vy / 20,
                    opponent.health / opponent_stats.max_health,
                    1.0 if opponent.weapon and opponent.weapon.is_attacking else 0.0
                ])
                
                # Distância das bordas (2 valores)
                border_x = min(
                    agent.x - self.game.arena_config.border,
                    SCREEN_WIDTH - self.game.arena_config.border - agent.x
                ) / (SCREEN_WIDTH / 2)
                border_y = min(
                    agent.y - self.game.arena_config.border,
                    SCREEN_HEIGHT - self.game.arena_config.border - agent.y
                ) / (SCREEN_HEIGHT / 2)
                
                obs.extend([border_x, border_y])
                
                return np.array(obs, dtype=np.float32)
            
            def decide_actions(self, observation):
                if self.neural_network is None:
                    return SimpleAI().decide_actions(observation)
                
                obs_vector = self._get_observation_19()
                
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_vector).unsqueeze(0)
                    action_mean, action_std, value = self.neural_network(obs_tensor)
                    action_vector = action_mean.squeeze().numpy()
                
                return self.actions_from_vector(action_vector)
        
        return NeuralGameAI(self.loaded_game_ai, game_ref)
    
    def start_selection(self, game_mode: str):
        """Inicia o processo de seleção de personagem"""
        self.game_mode_choice = game_mode
        self.current_player_selecting = 1
        self.selection_step = "class"
        self.mode = "select"
    
    def reset_game(self, mode: str = "pvp"):
        """Reseta o jogo para um novo round"""
        self.entities.clear()
        self.game_state = GameState(self.arena_config, self.game_config)
        self.game_over = False
        self.winner = None
        self.mode = mode
        
        # Obter classes e armas selecionadas
        class_p1 = self.get_selected_class_id(1)
        weapon_p1 = self.get_selected_weapon_id(1)
        class_p2 = self.get_selected_class_id(2)
        weapon_p2 = self.get_selected_weapon_id(2)
        
        # Configurar sistema de mapa grande se ativado
        if self.large_map_enabled:
            self._setup_large_map()
            # Posições em cantos opostos do mapa grande
            p1_x = 200
            p1_y = 200
            p2_x = self.large_map_width - 200
            p2_y = self.large_map_height - 200
        else:
            # Posições baseadas no tamanho da tela (mapa normal)
            p1_x = int(SCREEN_WIDTH * 0.25)
            p2_x = int(SCREEN_WIDTH * 0.75)
            p1_y = SCREEN_HEIGHT // 2
            p2_y = SCREEN_HEIGHT // 2
        
        if mode == "pvp":
            # Jogador 1
            p1_controller = PlayerController(
                pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d,
                pygame.K_SPACE, pygame.K_LSHIFT
            )
            player1 = self.create_entity(
                class_p1, p1_x, p1_y, (255, 100, 100), 
                p1_controller, weapon_p1
            )
            
            # Jogador 2
            p2_controller = PlayerController(
                pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                pygame.K_RETURN, pygame.K_RSHIFT
            )
            player2 = self.create_entity(
                class_p2, p2_x, p2_y, (100, 100, 255),
                p2_controller, weapon_p2
            )
            
            # Definir camera_target como player1 em mapa grande
            if self.large_map_enabled:
                self.camera_target = player1
        
        elif mode == "pve":
            # Jogador
            p1_controller = PlayerController(
                pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d,
                pygame.K_SPACE, pygame.K_LSHIFT
            )
            player = self.create_entity(
                class_p1, p1_x, p1_y, (100, 200, 100),
                p1_controller, weapon_p1
            )
            
            # IA - usar treinada se disponível
            if self.use_trained_ai and self.loaded_game_ai is not None:
                ai_controller = self._create_neural_ai_controller()
            else:
                # Usa StrategicAI com estratégia específica para a combinação
                ai_controller = StrategicAI(
                    class_name=class_p2.capitalize(),
                    weapon_name=weapon_p2.capitalize()
                )
            
            ai = self.create_entity(
                class_p2, p2_x, p2_y, (255, 100, 100),
                ai_controller, weapon_p2
            )
            
            # Configurar alvos da IA
            ai_controller.set_targets([player])
            
            # Definir camera_target como player em mapa grande
            if self.large_map_enabled:
                self.camera_target = player
        
        elif mode == "ai_vs_ai":
            # IA 1 - usar treinada se disponível
            if self.use_trained_ai and self.loaded_game_ai is not None:
                ai1_controller = self._create_neural_ai_controller()
            else:
                ai1_controller = StrategicAI(
                    class_name=class_p1.capitalize(),
                    weapon_name=weapon_p1.capitalize()
                )
            
            ai1 = self.create_entity(
                class_p1, p1_x, p1_y, (100, 200, 255),
                ai1_controller, weapon_p1
            )
            
            # IA 2 - StrategicAI com estratégia específica
            ai2_controller = StrategicAI(
                class_name=class_p2.capitalize(),
                weapon_name=weapon_p2.capitalize()
            )
            ai2 = self.create_entity(
                class_p2, p2_x, p2_y, (255, 200, 100),
                ai2_controller, weapon_p2
            )
            
            # Configurar alvos
            ai1_controller.set_targets([ai2])
            ai2_controller.set_targets([ai1])
            
            # Definir camera_target como ai1 em mapa grande
            if self.large_map_enabled:
                self.camera_target = ai1
        
        self.game_state.reset()
    
    def _setup_large_map(self):
        """Configura o sistema de mapa grande com fog of war"""
        # Criar câmera com dimensões da tela E do mapa
        self.camera = Camera(
            SCREEN_WIDTH, SCREEN_HEIGHT,
            self.large_map_width, self.large_map_height
        )
        
        # Criar gerenciador de obstáculos
        self.obstacle_manager = ObstacleManager()
        # Gerar obstáculos usando generate_for_map
        self.obstacle_manager.generate_for_map(
            map_width=self.large_map_width,
            map_height=self.large_map_height,
            map_type="large_arena",
            border=150,
            density=0.5
        )
        
        # Criar sistema de fog of war
        self.fog_of_war = FogOfWar(
            self.large_map_width,
            self.large_map_height
        )
        self.fog_of_war.set_obstacle_manager(self.obstacle_manager)
    
    def _check_projectiles_obstacles(self):
        """Verifica colisão de projéteis com obstáculos"""
        if not self.obstacle_manager:
            return
        
        for entity in self.entities:
            if not entity.is_alive():
                continue
            
            # Verificar flechas
            if hasattr(entity, 'weapon') and entity.weapon:
                if hasattr(entity.weapon, 'arrows'):
                    for arrow in entity.weapon.arrows:
                        if arrow.get('active', False):
                            arrow_rect = pygame.Rect(
                                arrow['x'] - 5, arrow['y'] - 5, 10, 10
                            )
                            for obs in self.obstacle_manager.obstacles:
                                if obs.rect.colliderect(arrow_rect):
                                    arrow['active'] = False
                                    break
    
    def handle_events(self):
        """Processa eventos do pygame"""
        global SCREEN_WIDTH, SCREEN_HEIGHT, screen, font, font_large, font_title, font_sizes
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.VIDEORESIZE:
                # Atualizar tamanho da tela
                SCREEN_WIDTH = max(1024, event.w)
                SCREEN_HEIGHT = max(700, event.h)
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                
                # Atualizar fontes
                font_sizes = get_font_sizes()
                font = pygame.font.Font(None, font_sizes['small'])
                font_large = pygame.font.Font(None, font_sizes['medium'])
                font_title = pygame.font.Font(None, font_sizes['large'])
                
                # Atualizar arena e física
                self.arena_config = ArenaConfig(SCREEN_WIDTH, SCREEN_HEIGHT, 50)
                self.physics = Physics(SCREEN_WIDTH, SCREEN_HEIGHT)
                
                # Atualizar mapa
                self.map_renderer.resize(SCREEN_WIDTH, SCREEN_HEIGHT)
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.mode == "menu":
                        self.running = False
                    elif self.mode == "select":
                        self.mode = "menu"
                    else:
                        self.mode = "menu"
                
                elif event.key == pygame.K_r and self.mode not in ["menu", "select"]:
                    self.reset_game(self.mode)
                
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                
                # Menu controls
                elif self.mode == "menu":
                    if event.key == pygame.K_1:
                        self.start_selection("pvp")
                    elif event.key == pygame.K_2:
                        self.start_selection("pve")
                    elif event.key == pygame.K_3:
                        self.start_selection("ai_vs_ai")
                    elif event.key == pygame.K_4:
                        self.mode = "training_menu"
                    elif event.key == pygame.K_5:
                        # Menu de batalha em grupo integrado
                        self.mode = "group_menu"
                        self.group_menu_state = "main"
                    elif event.key == pygame.K_6:
                        self.editor_mode = "classes"
                        self.editor_selected_item = 0
                        self.editor_selected_stat = 0
                        self.mode = "editor"
                
                # Menu de batalha em grupo
                elif self.mode == "group_menu":
                    self.handle_group_menu_input(event.key)
                
                # Batalha em grupo ativa
                elif self.mode == "group_battle":
                    self.handle_group_battle_input(event.key)
                
                # Editor de atributos
                elif self.mode == "editor":
                    self.handle_editor_input(event.key)
                
                # Menu de treinamento
                elif self.mode == "training_menu":
                    self.handle_training_menu_input(event.key)
                
                # Seleção para treinamento
                elif self.mode == "training_select":
                    self.handle_training_selection_input(event.key)
                
                # Treinamento ativo
                elif self.mode == "training":
                    self.handle_training_input(event.key)
                
                # Seleção de personagem
                elif self.mode == "select":
                    self.handle_selection_input(event.key)
    
    def handle_selection_input(self, key):
        """Processa input na tela de seleção"""
        player = self.current_player_selecting
        
        # Navegação
        if key in [pygame.K_LEFT, pygame.K_a]:
            if self.selection_step == "class":
                if player == 1:
                    self.selected_class_p1 = (self.selected_class_p1 - 1) % len(self.available_classes)
                else:
                    self.selected_class_p2 = (self.selected_class_p2 - 1) % len(self.available_classes)
            elif self.selection_step == "weapon":
                if player == 1:
                    self.selected_weapon_p1 = (self.selected_weapon_p1 - 1) % len(self.available_weapons)
                else:
                    self.selected_weapon_p2 = (self.selected_weapon_p2 - 1) % len(self.available_weapons)
            elif self.selection_step == "map":
                self.selected_map = (self.selected_map - 1) % len(self.available_maps)
            elif self.selection_step == "ai_model":
                if len(self.game_ai_models) > 0:
                    self.selected_ai_model_idx = (self.selected_ai_model_idx - 1) % (len(self.game_ai_models) + 1)
        
        elif key in [pygame.K_RIGHT, pygame.K_d]:
            if self.selection_step == "class":
                if player == 1:
                    self.selected_class_p1 = (self.selected_class_p1 + 1) % len(self.available_classes)
                else:
                    self.selected_class_p2 = (self.selected_class_p2 + 1) % len(self.available_classes)
            elif self.selection_step == "weapon":
                if player == 1:
                    self.selected_weapon_p1 = (self.selected_weapon_p1 + 1) % len(self.available_weapons)
                else:
                    self.selected_weapon_p2 = (self.selected_weapon_p2 + 1) % len(self.available_weapons)
            elif self.selection_step == "map":
                self.selected_map = (self.selected_map + 1) % len(self.available_maps)
            elif self.selection_step == "ai_model":
                if len(self.game_ai_models) > 0:
                    self.selected_ai_model_idx = (self.selected_ai_model_idx + 1) % (len(self.game_ai_models) + 1)
        
        # Toggle IA treinada com UP/DOWN no passo de seleção de modelo
        elif key in [pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s]:
            if self.selection_step == "ai_model":
                if len(self.game_ai_models) > 0:
                    self.selected_ai_model_idx = (self.selected_ai_model_idx + 1) % (len(self.game_ai_models) + 1)
        
        # Confirmar seleção
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            if self.selection_step == "class":
                self.selection_step = "weapon"
            elif self.selection_step == "weapon":
                # Arma selecionada, próximo jogador ou passo seguinte
                if self.game_mode_choice == "pve" and player == 1:
                    # No PvE, jogador 2 é IA - selecionar classe/arma da IA
                    self.current_player_selecting = 2
                    self.selection_step = "class"
                elif player == 1:
                    self.current_player_selecting = 2
                    self.selection_step = "class"
                else:
                    # Ambos selecionaram, ir para seleção de mapa
                    self.selection_step = "map"
            elif self.selection_step == "map":
                # Mapa selecionado
                self.current_map = self.available_maps[self.selected_map]
                # Se for PvE ou AI vs AI, ir para seleção de modelo IA
                if self.game_mode_choice in ["pve", "ai_vs_ai"]:
                    self._load_game_ai_models()
                    self.selection_step = "ai_model"
                else:
                    # PvP - iniciar jogo direto
                    self.reset_game(self.game_mode_choice)
            elif self.selection_step == "ai_model":
                # Modelo de IA selecionado, carregar e iniciar jogo
                self._apply_selected_ai_model()
                self.reset_game(self.game_mode_choice)
        
        # Toggle mapa grande com Fog of War
        elif key == pygame.K_l:
            if self.selection_step == "map":
                self.large_map_enabled = not self.large_map_enabled
        
        # Voltar
        elif key == pygame.K_BACKSPACE:
            if self.selection_step == "weapon":
                self.selection_step = "class"
            elif self.selection_step == "map":
                self.current_player_selecting = 2
                self.selection_step = "weapon"
            elif self.selection_step == "ai_model":
                self.selection_step = "map"
            elif player == 2 and self.selection_step == "class":
                self.current_player_selecting = 1
                self.selection_step = "weapon"
    
    # ==========================================================================
    # SISTEMA DE BATALHA EM GRUPO INTEGRADO
    # ==========================================================================
    
    def handle_group_menu_input(self, key):
        """Processa input no menu de batalha em grupo"""
        if key == pygame.K_ESCAPE:
            if self.group_menu_state == "main":
                self.mode = "menu"
            elif self.group_menu_state in ["select_size", "select_blue", "select_red"]:
                self.group_menu_state = "main"
            return
        
        if self.group_menu_state == "main":
            if key == pygame.K_1:
                # Batalha rápida
                self.group_menu_state = "select_size"
            elif key == pygame.K_2:
                # Treinar em grupo
                self.group_menu_state = "train_setup"
            elif key == pygame.K_3:
                # Ver estatísticas
                self.group_menu_state = "stats"
        
        elif self.group_menu_state == "select_size":
            if key == pygame.K_1:
                self.team_size = 2
                self.group_menu_state = "select_blue"
                self.group_selecting_team = "blue"
                self.group_selected_composition = 0
            elif key == pygame.K_2:
                self.team_size = 3
                self.group_menu_state = "select_blue"
                self.group_selecting_team = "blue"
                self.group_selected_composition = 0
            elif key == pygame.K_3:
                self.team_size = 5
                self.group_menu_state = "select_blue"
                self.group_selecting_team = "blue"
                self.group_selected_composition = 0
        
        elif self.group_menu_state in ["select_blue", "select_red"]:
            compositions = self._get_compositions_for_size(self.team_size)
            comp_keys = list(compositions.keys())
            
            if key in [pygame.K_UP, pygame.K_w]:
                self.group_selected_composition = (self.group_selected_composition - 1) % len(comp_keys)
            elif key in [pygame.K_DOWN, pygame.K_s]:
                self.group_selected_composition = (self.group_selected_composition + 1) % len(comp_keys)
            elif key in [pygame.K_RETURN, pygame.K_SPACE]:
                selected_comp = compositions[comp_keys[self.group_selected_composition]]
                
                if self.group_menu_state == "select_blue":
                    self.blue_team_config = selected_comp.copy()
                    self.group_menu_state = "select_red"
                    self.group_selecting_team = "red"
                    self.group_selected_composition = 0
                else:
                    self.red_team_config = selected_comp.copy()
                    self._start_group_battle()
        
        elif self.group_menu_state == "stats":
            if key == pygame.K_ESCAPE or key == pygame.K_BACKSPACE:
                self.group_menu_state = "main"
    
    def _get_compositions_for_size(self, size):
        """Retorna composições disponíveis para o tamanho de equipe"""
        prefix = f"{size}v{size}_"
        return {k: v for k, v in self.team_compositions.items() if k.startswith(prefix)}
    
    def _start_group_battle(self):
        """Inicia uma batalha em grupo"""
        self.mode = "group_battle"
        self.team_battle_active = True
        self.paused = False
        
        # Limpar entidades anteriores
        self.entities.clear()
        self.blue_team.clear()
        self.red_team.clear()
        self.game_state = GameState(self.arena_config, self.game_config)
        
        # Posições de spawn
        margin = 100
        blue_x_base = int(SCREEN_WIDTH * 0.2)
        red_x_base = int(SCREEN_WIDTH * 0.8)
        
        # Calcular espaçamento vertical
        y_spacing = (SCREEN_HEIGHT - 2 * margin) // max(1, self.team_size - 1) if self.team_size > 1 else 0
        center_y = SCREEN_HEIGHT // 2
        
        # Criar time azul
        for i, config in enumerate(self.blue_team_config):
            y = margin + i * y_spacing if self.team_size > 1 else center_y
            x = blue_x_base + random.randint(-30, 30)
            
            # Criar controlador IA com estratégia baseada no papel
            ai_controller = StrategicAI(
                class_name=config["class"].capitalize(),
                weapon_name=config["weapon"].capitalize()
            )
            
            entity = self.create_entity(
                config["class"], x, y, (80, 140, 255),
                ai_controller, config["weapon"]
            )
            entity._role = config.get("role", "dps_melee")
            entity._team = "blue"
            entity._prev_health = entity.health
            self.blue_team.append(entity)
        
        # Criar time vermelho
        for i, config in enumerate(self.red_team_config):
            y = margin + i * y_spacing if self.team_size > 1 else center_y
            x = red_x_base + random.randint(-30, 30)
            
            ai_controller = StrategicAI(
                class_name=config["class"].capitalize(),
                weapon_name=config["weapon"].capitalize()
            )
            
            entity = self.create_entity(
                config["class"], x, y, (255, 80, 80),
                ai_controller, config["weapon"]
            )
            entity._role = config.get("role", "dps_melee")
            entity._team = "red"
            entity._prev_health = entity.health
            self.red_team.append(entity)
        
        # Configurar alvos
        for entity in self.blue_team:
            if entity.controller:
                entity.controller.set_targets(self.red_team)
        
        for entity in self.red_team:
            if entity.controller:
                entity.controller.set_targets(self.blue_team)
        
        # Configurar aliados para habilidades de suporte (Cleric, Enchanter, etc.)
        for entity in self.blue_team:
            entity.set_allies(self.blue_team)
        for entity in self.red_team:
            entity.set_allies(self.red_team)
        
        self.game_state.reset()
        self.game_over = False
        self.winner = None
    
    def handle_group_battle_input(self, key):
        """Processa input durante batalha em grupo"""
        if key == pygame.K_ESCAPE:
            self._end_group_battle()
        elif key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_r:
            # Reiniciar batalha
            self._start_group_battle()
        elif key == pygame.K_1:
            self.training_speed = 1
        elif key == pygame.K_2:
            self.training_speed = 2
        elif key == pygame.K_4:
            self.training_speed = 4
    
    def _end_group_battle(self):
        """Finaliza a batalha em grupo"""
        self.team_battle_active = False
        self.mode = "group_menu"
        self.group_menu_state = "main"
        
        # Limpar
        self.entities.clear()
        self.blue_team.clear()
        self.red_team.clear()
    
    def _update_group_battle(self, dt):
        """Atualiza a batalha em grupo"""
        if not self.team_battle_active or self.paused:
            return
        
        # Executar múltiplos steps se velocidade > 1
        for _ in range(self.training_speed):
            # Atualizar todas as entidades
            for entity in self.entities:
                if entity.is_alive():
                    entity.update(dt)
            
            # Verificar armadilhas do Trapper
            for entity in self.entities:
                if hasattr(entity, 'check_traps') and entity.is_alive():
                    enemies = [e for e in self.entities if e != entity and e.is_alive() and (entity.team == "none" or e.team != entity.team)]
                    entity.check_traps(enemies)
            
            # Verificar armadilhas do TrapLauncher (arma)
            for entity in self.entities:
                if entity.is_alive() and hasattr(entity, 'weapon') and entity.weapon:
                    if hasattr(entity.weapon, 'check_trap_hits'):
                        hits = entity.weapon.check_trap_hits(self.entities)
                        for target, trap in hits:
                            target.take_damage(trap['damage'], entity)
                            from stats import StatusEffect, StatusEffectType
                            target.apply_status_effect(StatusEffect(
                                name='launcher_trap_root',
                                effect_type=StatusEffectType.ROOT,
                                duration=trap['root_duration'],
                                source=entity
                            ))
            
            # Verificar armas especiais (Staff heal, Tome buff)
            self._check_special_weapons()
            
            # Física
            self.physics.handle_collisions(self.entities)
            
            # Limitar à arena
            arena_rect = pygame.Rect(*self.arena_config.playable_rect)
            for entity in self.entities:
                self.physics.constrain_to_arena(entity, arena_rect)
            
            # Colisão de projéteis com bordas da arena
            self.physics.check_projectiles_arena_collision(self.entities, arena_rect)
            
            # Verificar fim da batalha
            blue_alive = sum(1 for e in self.blue_team if e.is_alive())
            red_alive = sum(1 for e in self.red_team if e.is_alive())
            
            if blue_alive == 0 or red_alive == 0:
                self.game_over = True
                self.group_battle_stats["total_battles"] += 1
                
                if blue_alive > 0:
                    self.winner = "blue"
                    self.group_battle_stats["blue_wins"] += 1
                elif red_alive > 0:
                    self.winner = "red"
                    self.group_battle_stats["red_wins"] += 1
                else:
                    self.winner = "draw"
                    self.group_battle_stats["draws"] += 1
                
                self.team_battle_active = False
                break
    
    def _check_special_weapons(self):
        """Verifica e aplica efeitos de armas especiais (Staff cura, Tome buffa)"""
        from stats import StatusEffect, StatusEffectType
        import math
        
        for entity in self.entities:
            if not entity.is_alive() or not hasattr(entity, 'weapon') or not entity.weapon:
                continue
            
            weapon = entity.weapon
            if not weapon.is_attacking:
                continue
            
            hitbox = weapon.get_hitbox()
            if not hitbox:
                continue
            
            # Staff de Cura - cura aliados próximos
            if hitbox.get('is_heal_weapon', False):
                heal_info = hitbox.get('heal_info', {})
                if heal_info.get('active', False):
                    heal_amount = heal_info.get('amount', 10)
                    heal_range = heal_info.get('range', 100)
                    
                    # Cura a si mesmo (metade)
                    entity.heal(heal_amount * 0.5)
                    
                    # Cura aliados no raio
                    for other in self.entities:
                        if other == entity or not other.is_alive():
                            continue
                        # Verificar se é aliado (mesmo team ou sem team definido)
                        is_ally = (entity.team != "none" and other.team == entity.team) or \
                                  (entity.team == "none" and other in getattr(entity, 'allies', []))
                        if is_ally:
                            dx = other.x - entity.x
                            dy = other.y - entity.y
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist <= heal_range:
                                other.heal(heal_amount)
            
            # Tome - buffa aliados próximos
            if hitbox.get('is_buff_weapon', False):
                buff_effect = hitbox.get('buff_effect')
                buff_range = hitbox.get('buff_range', 100)
                
                if buff_effect:
                    # Buffa a si mesmo
                    entity.apply_status_effect(buff_effect)
                    
                    # Buffa aliados
                    for other in self.entities:
                        if other == entity or not other.is_alive():
                            continue
                        is_ally = (entity.team != "none" and other.team == entity.team) or \
                                  (entity.team == "none" and other in getattr(entity, 'allies', []))
                        if is_ally:
                            dx = other.x - entity.x
                            dy = other.y - entity.y
                            dist = math.sqrt(dx*dx + dy*dy)
                            if dist <= buff_range:
                                # Criar nova instância do efeito para cada aliado
                                from stats import StatusEffect
                                ally_buff = StatusEffect(
                                    name=buff_effect.name,
                                    effect_type=buff_effect.effect_type,
                                    duration=buff_effect.duration,
                                    power=buff_effect.power,
                                    source=entity
                                )
                                other.apply_status_effect(ally_buff)
    
    def _calculate_group_reward(self, entity, role, allies, enemies, prev_health):
        """Calcula recompensa baseada no papel do agente no grupo"""
        reward = 0.0
        weights = self.role_reward_weights.get(role, self.role_reward_weights["dps_melee"])
        
        # Dano recebido
        damage_taken = prev_health - entity.health
        if damage_taken > 0:
            reward += damage_taken * weights.get("damage_taken", -0.02)
        
        # Sobrevivência
        if entity.is_alive():
            reward += 0.001
        else:
            reward += weights.get("death", -1.0)
        
        # Recompensas específicas por papel
        if role == "healer":
            # Bônus por aliados vivos com boa vida
            for ally in allies:
                if ally.is_alive() and ally != entity:
                    health_pct = ally.health / ally.stats_manager.get_stats().max_health
                    reward += health_pct * weights.get("allies_alive", 0.02)
        
        elif role == "tank":
            # Bônus por estar na frente (entre aliados e inimigos)
            for enemy in enemies:
                if enemy.is_alive():
                    dist_to_enemy = math.sqrt((entity.x - enemy.x)**2 + (entity.y - enemy.y)**2)
                    for ally in allies:
                        if ally != entity and ally.is_alive():
                            ally_dist = math.sqrt((ally.x - enemy.x)**2 + (ally.y - enemy.y)**2)
                            if dist_to_enemy < ally_dist:
                                reward += weights.get("allies_protected", 0.02)
                                break
        
        elif role in ["dps_melee", "dps_ranged"]:
            # Bônus por focar inimigos com pouca vida
            for enemy in enemies:
                if enemy.is_alive():
                    health_pct = enemy.health / enemy.stats_manager.get_stats().max_health
                    if health_pct < 0.3:
                        dist = math.sqrt((entity.x - enemy.x)**2 + (entity.y - enemy.y)**2)
                        if dist < 200:
                            reward += 0.02
        
        return reward

    def handle_training_menu_input(self, key):
        """Processa input no menu de treinamento"""
        if self.model_stats_view_active:
            # Visualizando estatísticas de modelo
            if key == pygame.K_ESCAPE or key == pygame.K_BACKSPACE:
                self.model_stats_view_active = False
                # Permanecer no menu de treinamento, não ir para o menu principal
                return
            elif key in [pygame.K_UP, pygame.K_w]:
                if len(self.available_models) > 0:
                    self.model_stats_selected = (self.model_stats_selected - 1) % len(self.available_models)
            elif key in [pygame.K_DOWN, pygame.K_s]:
                if len(self.available_models) > 0:
                    self.model_stats_selected = (self.model_stats_selected + 1) % len(self.available_models)
            elif key == pygame.K_DELETE or key == pygame.K_x:
                # Excluir modelo selecionado
                self._delete_selected_model()
            return  # Não processar outras teclas enquanto estiver na visualização
        elif self.load_menu_active:
            # Navegando no menu de carregar modelos
            if key == pygame.K_ESCAPE or key == pygame.K_BACKSPACE:
                self.load_menu_active = False
            elif key in [pygame.K_UP, pygame.K_w]:
                if len(self.available_models) > 0:
                    self.selected_model_idx = (self.selected_model_idx - 1) % len(self.available_models)
            elif key in [pygame.K_DOWN, pygame.K_s]:
                if len(self.available_models) > 0:
                    self.selected_model_idx = (self.selected_model_idx + 1) % len(self.available_models)
            elif key in [pygame.K_RETURN, pygame.K_SPACE]:
                if len(self.available_models) > 0:
                    self.load_selected_model()
            elif key == pygame.K_DELETE or key == pygame.K_x:
                # Excluir modelo selecionado
                self._delete_selected_model()
        else:
            if key == pygame.K_ESCAPE:
                self.mode = "menu"
            elif key == pygame.K_1:
                # Novo treinamento
                self.training_selection_step = "agent_class"
                self.mode = "training_select"
            elif key == pygame.K_2:
                # Carregar modelo - mostrar lista
                self.load_trained_model()
            elif key == pygame.K_3:
                # Testar modelo
                self.test_trained_model()
            elif key == pygame.K_4:
                # Ver estatísticas dos modelos
                self._load_models_for_stats()
                self.model_stats_view_active = True
    
    def _load_models_for_stats(self):
        """Carrega modelos para visualização de estatísticas"""
        import json
        self.available_models = []
        self.model_stats_selected = 0
        
        if os.path.exists("models"):
            files = [f for f in os.listdir("models") if f.endswith('.json')]
            for f in files:
                try:
                    with open(os.path.join("models", f), 'r') as file:
                        data = json.load(file)
                        data['filename'] = f
                        self.available_models.append(data)
                except:
                    pass
    
    def _delete_selected_model(self):
        """Exclui o modelo selecionado"""
        if not self.available_models:
            return
        
        idx = self.model_stats_selected if self.model_stats_view_active else self.selected_model_idx
        model = self.available_models[idx]
        
        try:
            # Deletar arquivo JSON
            json_path = os.path.join("models", model['filename'])
            if os.path.exists(json_path):
                os.remove(json_path)
            
            # Deletar arquivo de pesos se existir
            if model.get('weights_file'):
                weights_path = os.path.join("models", model['weights_file'])
                if os.path.exists(weights_path):
                    os.remove(weights_path)
            
            # Remover da lista
            self.available_models.pop(idx)
            
            # Ajustar índice selecionado
            if self.model_stats_view_active:
                if self.model_stats_selected >= len(self.available_models):
                    self.model_stats_selected = max(0, len(self.available_models) - 1)
            else:
                if self.selected_model_idx >= len(self.available_models):
                    self.selected_model_idx = max(0, len(self.available_models) - 1)
            
            self.save_message = "🗑️ Modelo excluído com sucesso!"
            self.save_message_timer = 2.0
        except Exception as e:
            self.save_message = f"❌ Erro ao excluir: {str(e)}"
            self.save_message_timer = 3.0
    
    def handle_editor_input(self, key):
        """Processa input no editor de atributos"""
        if key == pygame.K_ESCAPE:
            self.mode = "menu"
        elif key == pygame.K_TAB:
            # Alternar entre classes e armas
            self.editor_mode = "weapons" if self.editor_mode == "classes" else "classes"
            self.editor_selected_item = 0
            self.editor_selected_stat = 0
        elif key in [pygame.K_LEFT, pygame.K_a]:
            # Item anterior
            items = self.available_classes if self.editor_mode == "classes" else self.available_weapons
            self.editor_selected_item = (self.editor_selected_item - 1) % len(items)
            self.editor_selected_stat = 0
        elif key in [pygame.K_RIGHT, pygame.K_d]:
            # Próximo item
            items = self.available_classes if self.editor_mode == "classes" else self.available_weapons
            self.editor_selected_item = (self.editor_selected_item + 1) % len(items)
            self.editor_selected_stat = 0
        elif key in [pygame.K_UP, pygame.K_w]:
            # Stat anterior
            stats = self._get_current_editor_stats()
            self.editor_selected_stat = (self.editor_selected_stat - 1) % len(stats)
        elif key in [pygame.K_DOWN, pygame.K_s]:
            # Próximo stat
            stats = self._get_current_editor_stats()
            self.editor_selected_stat = (self.editor_selected_stat + 1) % len(stats)
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS or key == pygame.K_KP_PLUS:
            # Aumentar valor
            self._modify_editor_stat(1)
        elif key == pygame.K_MINUS or key == pygame.K_KP_MINUS:
            # Diminuir valor
            self._modify_editor_stat(-1)
        elif key == pygame.K_r:
            # Resetar para padrão
            self._reset_editor_stat()
    
    def _get_current_editor_stats(self):
        """Retorna as stats do item atualmente selecionado no editor (do banco de dados central)"""
        items = self.available_classes if self.editor_mode == "classes" else self.available_weapons
        item_id = items[self.editor_selected_item]
        
        if self.editor_mode == "classes":
            return list(self.config_db.get_class_stats(item_id).items())
        else:
            return list(self.config_db.get_weapon_stats(item_id).items())
    
    def _modify_editor_stat(self, direction: int):
        """Modifica o valor do stat selecionado no banco de dados central"""
        items = self.available_classes if self.editor_mode == "classes" else self.available_weapons
        item_id = items[self.editor_selected_item]
        
        if self.editor_mode == "classes":
            stats = self.config_db.get_class_stats(item_id)
        else:
            stats = self.config_db.get_weapon_stats(item_id)
        
        stat_keys = list(stats.keys())
        stat_key = stat_keys[self.editor_selected_stat]
        
        # Definir incrementos por tipo de stat
        increments = {
            'max_health': 10,
            'speed': 0.25,
            'acceleration': 0.05,
            'defense': 1,
            'damage_multiplier': 0.1,
            'base_damage': 2,
            'range': 5,
            'attack_cooldown': 0.05,
            'knockback_force': 1,
            'critical_chance': 0.05
        }
        
        increment = increments.get(stat_key, 1) * direction
        new_value = stats[stat_key] + increment
        
        # Limites mínimos
        mins = {
            'max_health': 10,
            'speed': 0.5,
            'acceleration': 0.1,
            'defense': 0,
            'damage_multiplier': 0.1,
            'base_damage': 1,
            'range': 20,
            'attack_cooldown': 0.1,
            'knockback_force': 0,
            'critical_chance': 0
        }
        
        # Aplicar novo valor com limite mínimo
        new_value = max(mins.get(stat_key, 0), new_value)
        
        # Salvar no banco de dados central
        if self.editor_mode == "classes":
            self.config_db.set_class_stat(item_id, stat_key, new_value)
        else:
            self.config_db.set_weapon_stat(item_id, stat_key, new_value)
    
    def _reset_editor_stat(self):
        """Reseta todos os stats do item atual para os valores padrão"""
        items = self.available_classes if self.editor_mode == "classes" else self.available_weapons
        item_id = items[self.editor_selected_item]
        
        if self.editor_mode == "classes":
            self.config_db.reset_class_stats(item_id)
        else:
            self.config_db.reset_weapon_stats(item_id)
    
    def handle_training_selection_input(self, key):
        """Processa input na seleção de treinamento"""
        # Navegação
        if key in [pygame.K_LEFT, pygame.K_a]:
            if self.training_selection_step == "agent_class":
                self.training_class = (self.training_class - 1) % len(self.available_classes)
            elif self.training_selection_step == "agent_weapon":
                self.training_weapon = (self.training_weapon - 1) % len(self.available_weapons)
            elif self.training_selection_step == "opponent_type":
                self.training_multi_opponent = not self.training_multi_opponent
            elif self.training_selection_step == "opponent_class":
                self.training_opponent_class = (self.training_opponent_class - 1) % len(self.available_classes)
            elif self.training_selection_step == "opponent_weapon":
                self.training_opponent_weapon = (self.training_opponent_weapon - 1) % len(self.available_weapons)
            elif self.training_selection_step == "config":
                pass  # Usar UP/DOWN para config
        
        elif key in [pygame.K_RIGHT, pygame.K_d]:
            if self.training_selection_step == "agent_class":
                self.training_class = (self.training_class + 1) % len(self.available_classes)
            elif self.training_selection_step == "agent_weapon":
                self.training_weapon = (self.training_weapon + 1) % len(self.available_weapons)
            elif self.training_selection_step == "opponent_type":
                self.training_multi_opponent = not self.training_multi_opponent
            elif self.training_selection_step == "opponent_class":
                self.training_opponent_class = (self.training_opponent_class + 1) % len(self.available_classes)
            elif self.training_selection_step == "opponent_weapon":
                self.training_opponent_weapon = (self.training_opponent_weapon + 1) % len(self.available_weapons)
        
        elif key in [pygame.K_UP, pygame.K_w]:
            if self.training_selection_step == "config":
                self.training_episodes = min(100000, self.training_episodes + 1000)
        
        elif key in [pygame.K_DOWN, pygame.K_s]:
            if self.training_selection_step == "config":
                self.training_episodes = max(1000, self.training_episodes - 1000)
        
        # Confirmar
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            if self.training_selection_step == "agent_class":
                self.training_selection_step = "agent_weapon"
            elif self.training_selection_step == "agent_weapon":
                self.training_selection_step = "opponent_type"
            elif self.training_selection_step == "opponent_type":
                if self.training_multi_opponent:
                    # Pular seleção individual, vai usar todos
                    self._build_opponent_pool()
                    self.training_selection_step = "config"
                else:
                    self.training_selection_step = "opponent_class"
            elif self.training_selection_step == "opponent_class":
                self.training_selection_step = "opponent_weapon"
            elif self.training_selection_step == "opponent_weapon":
                self.training_selection_step = "config"
            elif self.training_selection_step == "config":
                self.start_training()
        
        # Voltar
        elif key == pygame.K_BACKSPACE:
            if self.training_selection_step == "agent_weapon":
                self.training_selection_step = "agent_class"
            elif self.training_selection_step == "opponent_type":
                self.training_selection_step = "agent_weapon"
            elif self.training_selection_step == "opponent_class":
                self.training_selection_step = "opponent_type"
            elif self.training_selection_step == "opponent_weapon":
                self.training_selection_step = "opponent_class"
            elif self.training_selection_step == "config":
                if self.training_multi_opponent:
                    self.training_selection_step = "opponent_type"
                else:
                    self.training_selection_step = "opponent_weapon"
            else:
                self.mode = "training_menu"
        
        elif key == pygame.K_ESCAPE:
            self.mode = "training_menu"
    
    def _build_opponent_pool(self):
        """Constrói o pool de oponentes para treinamento variado"""
        self.training_opponent_pool = []
        # Adiciona todas as combinações de classe + arma
        for cls_id in self.available_classes:
            for wpn_id in self.available_weapons:
                self.training_opponent_pool.append((cls_id, wpn_id))
        # Embaralha para variedade
        random.shuffle(self.training_opponent_pool)
        self.current_opponent_idx = 0
    
    def _get_current_opponent_config(self):
        """Retorna a configuração do oponente atual (classe, arma)"""
        if self.training_multi_opponent and self.training_opponent_pool:
            return self.training_opponent_pool[self.current_opponent_idx]
        else:
            return (
                self.available_classes[self.training_opponent_class],
                self.available_weapons[self.training_opponent_weapon]
            )
    
    def _rotate_opponent(self):
        """Troca para o próximo oponente no pool"""
        if self.training_multi_opponent and self.training_opponent_pool:
            self.current_opponent_idx = (self.current_opponent_idx + 1) % len(self.training_opponent_pool)
    
    def handle_training_input(self, key):
        """Processa input durante o treinamento"""
        if key == pygame.K_ESCAPE:
            self.stop_training()
            self.mode = "training_menu"
        elif key == pygame.K_SPACE:
            self.training_paused = not self.training_paused
        elif key == pygame.K_v:
            self.training_render = not self.training_render
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.training_speed = min(10, self.training_speed + 1)
        elif key == pygame.K_MINUS:
            self.training_speed = max(0, self.training_speed - 1)
        elif key == pygame.K_s:
            self.save_current_model()
    
    def start_training(self):
        """Inicia o treinamento da IA com rede neural"""
        self.mode = "training"
        self.training_running = True
        self.training_paused = False
        self.training_step_count = 0
        
        self.training_stats = {
            'episode': 0,
            'total_episodes': self.training_episodes,
            'wins': 0,
            'losses': 0,
            'avg_reward': 0,
            'recent_rewards': deque(maxlen=100),
            'best_reward': float('-inf'),
            'training_time': time.time(),
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'learning_updates': 0
        }
        
        # Obter IDs de classe/arma
        agent_class = self.available_classes[self.training_class]
        agent_weapon = self.available_weapons[self.training_weapon]
        
        # Obter configuração do oponente (único ou primeiro do pool)
        if self.training_multi_opponent:
            if not self.training_opponent_pool:
                self._build_opponent_pool()
            opponent_class, opponent_weapon = self._get_current_opponent_config()
        else:
            opponent_class = self.available_classes[self.training_opponent_class]
            opponent_weapon = self.available_weapons[self.training_opponent_weapon]
        
        # Criar pasta de modelos
        os.makedirs("models", exist_ok=True)
        
        # Inicializar rede neural se PyTorch disponível
        if TORCH_AVAILABLE:
            self.neural_network = ActorCritic(
                obs_size=self.nn_obs_size,
                action_size=self.nn_action_size,
                hidden_size=256
            )
            self.nn_optimizer = torch.optim.Adam(
                self.neural_network.parameters(), 
                lr=3e-4
            )
            print(f"🧠 Rede Neural inicializada!")
            print(f"   Arquitetura: {self.nn_obs_size} → 256 → 256 → {self.nn_action_size}")
        else:
            self.neural_network = None
            print("⚠️ Treinamento sem rede neural (PyTorch não disponível)")
        
        # Limpar buffer de experiência
        self._clear_experience_buffer()
        
        # Iniciar treinamento visual
        self.reset_training_episode(agent_class, agent_weapon, opponent_class, opponent_weapon)
    
    def _clear_experience_buffer(self):
        """Limpa o buffer de experiência"""
        self.experience_buffer = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def reset_training_episode(self, agent_class, agent_weapon, opponent_class, opponent_weapon):
        """Reseta para um novo episódio de treinamento"""
        self.entities.clear()
        self.game_state = GameState(self.arena_config, self.game_config)
        self.game_over = False
        
        # Posições aleatórias
        margin = 150
        agent_x = random.randint(margin, SCREEN_WIDTH // 2 - 50)
        agent_y = random.randint(margin, SCREEN_HEIGHT - margin)
        opponent_x = random.randint(SCREEN_WIDTH // 2 + 50, SCREEN_WIDTH - margin)
        opponent_y = random.randint(margin, SCREEN_HEIGHT - margin)
        
        # Criar agente (controlado pela rede neural)
        self.training_entity = self.create_entity(
            agent_class, agent_x, agent_y, (100, 255, 100),
            None, agent_weapon
        )
        
        # Criar oponente (StrategicAI com estratégia específica para a combinação)
        ai_controller = StrategicAI(
            class_name=opponent_class.capitalize(),
            weapon_name=opponent_weapon.capitalize()
        )
        self.training_opponent = self.create_entity(
            opponent_class, opponent_x, opponent_y, (255, 100, 100),
            ai_controller, opponent_weapon
        )
        ai_controller.set_targets([self.training_entity])
        
        self.training_step = 0
        self.training_episode_reward = 0
        self._prev_agent_health = self.training_entity.health
        self._prev_opponent_health = self.training_opponent.health
        self.game_state.reset()
    
    def _get_observation(self) -> np.ndarray:
        """Retorna a observação atual normalizada para a rede neural"""
        if not self.training_entity or not self.training_opponent:
            return np.zeros(self.nn_obs_size, dtype=np.float32)
        
        agent = self.training_entity
        opponent = self.training_opponent
        agent_stats = agent.stats_manager.get_stats()
        opponent_stats = opponent.stats_manager.get_stats()
        
        obs = []
        
        # Estado do agente (9 valores)
        obs.extend([
            agent.x / SCREEN_WIDTH,                                    # Posição X normalizada
            agent.y / SCREEN_HEIGHT,                                   # Posição Y normalizada
            agent.vx / 20,                                             # Velocidade X
            agent.vy / 20,                                             # Velocidade Y
            agent.facing_angle / math.pi,                              # Ângulo de face
            agent.health / agent_stats.max_health,                     # Vida normalizada
            1.0 if agent.invulnerable_time > 0 else 0.0,              # Invulnerável?
            1.0 if agent.weapon and agent.weapon.can_attack else 0.0, # Pode atacar?
            1.0 if agent.ability_cooldown <= 0 else 0.0               # Habilidade pronta?
        ])
        
        # Estado do oponente relativo (8 valores)
        rel_x = (opponent.x - agent.x) / SCREEN_WIDTH
        rel_y = (opponent.y - agent.y) / SCREEN_HEIGHT
        distance = math.sqrt(rel_x**2 + rel_y**2)
        angle_to = math.atan2(rel_y, rel_x)
        
        obs.extend([
            rel_x,                                                     # Posição relativa X
            rel_y,                                                     # Posição relativa Y
            distance,                                                  # Distância
            angle_to / math.pi,                                        # Ângulo até oponente
            opponent.vx / 20,                                          # Velocidade X do oponente
            opponent.vy / 20,                                          # Velocidade Y do oponente
            opponent.health / opponent_stats.max_health,               # Vida do oponente
            1.0 if opponent.weapon and opponent.weapon.is_attacking else 0.0  # Oponente atacando?
        ])
        
        # Distância das bordas (2 valores)
        border_x = min(
            agent.x - self.arena_config.border,
            SCREEN_WIDTH - self.arena_config.border - agent.x
        ) / (SCREEN_WIDTH / 2)
        border_y = min(
            agent.y - self.arena_config.border,
            SCREEN_HEIGHT - self.arena_config.border - agent.y
        ) / (SCREEN_HEIGHT / 2)
        
        obs.extend([border_x, border_y])
        
        return np.array(obs, dtype=np.float32)
    
    def update_training(self, dt: float):
        """Atualiza um passo do treinamento com rede neural"""
        if self.training_paused or not self.training_running:
            return
        
        # Velocidade de treinamento
        steps_per_frame = self.training_speed if self.training_speed > 0 else 10
        
        for _ in range(steps_per_frame):
            if self.game_over:
                break
            
            # Se está em slow motion, pular frames para dar o efeito
            if self.slow_motion and self.training_render:
                self.slow_motion_timer -= 1/60
                if self.slow_motion_timer <= 0:
                    self.slow_motion = False
                effective_dt = (1/60) * self.slow_motion_scale
            else:
                effective_dt = 1/60
            
            # Guardar vida anterior para detectar dano
            agent_health_before = self.training_entity.health if self.training_entity else 0
            opponent_health_before = self.training_opponent.health if self.training_opponent else 0
            
            # === PASSO DA REDE NEURAL ===
            # 1. Obter observação
            obs = self._get_observation()
            
            # 2. Obter ação da rede neural
            action, value, log_prob = self._get_neural_action(obs)
            
            # 3. Aplicar ação
            self._apply_neural_action(action)
            
            # Atualizar entidades
            for entity in self.entities:
                if entity.is_alive():
                    entity.update(effective_dt)
            
            # Verificar armadilhas do Trapper
            for entity in self.entities:
                if hasattr(entity, 'check_traps') and entity.is_alive():
                    enemies = [e for e in self.entities if e != entity and e.is_alive() and (entity.team == "none" or e.team != entity.team)]
                    entity.check_traps(enemies)
            
            # Verificar armadilhas do TrapLauncher (arma)
            for entity in self.entities:
                if entity.is_alive() and hasattr(entity, 'weapon') and entity.weapon:
                    if hasattr(entity.weapon, 'check_trap_hits'):
                        hits = entity.weapon.check_trap_hits(self.entities)
                        for target, trap in hits:
                            target.take_damage(trap['damage'], entity)
                            from stats import StatusEffect, StatusEffectType
                            target.apply_status_effect(StatusEffect(
                                name='launcher_trap_root',
                                effect_type=StatusEffectType.ROOT,
                                duration=trap['root_duration'],
                                source=entity
                            ))
            
            # Verificar armas especiais (Staff heal, Tome buff)
            self._check_special_weapons()
            
            # Física
            self.physics.handle_collisions(self.entities)
            
            # Arena
            arena_rect = pygame.Rect(*self.arena_config.playable_rect)
            for entity in self.entities:
                self.physics.constrain_to_arena(entity, arena_rect)
            
            # Colisão de projéteis com bordas da arena
            self.physics.check_projectiles_arena_collision(self.entities, arena_rect)
            
            # Detectar dano para ativar slow motion
            agent_took_damage = self.training_entity and self.training_entity.health < agent_health_before
            opponent_took_damage = self.training_opponent and self.training_opponent.health < opponent_health_before
            
            if (agent_took_damage or opponent_took_damage) and self.training_render:
                self.slow_motion = True
                self.slow_motion_timer = self.slow_motion_duration
                # Não break aqui para continuar acumulando experiência
            
            self.training_step += 1
            self.training_step_count += 1
            
            # 4. Calcular reward
            reward = self._calculate_training_reward()
            self.training_episode_reward += reward
            
            # 5. Verificar se episódio terminou
            terminated = False
            alive = [e for e in self.entities if e.is_alive()]
            if len(alive) <= 1 or self.training_step > 3000:
                self.game_over = True
                terminated = True
            
            # 6. Armazenar experiência
            if TORCH_AVAILABLE and self.neural_network:
                self.experience_buffer['obs'].append(obs)
                self.experience_buffer['actions'].append(action)
                self.experience_buffer['rewards'].append(reward)
                self.experience_buffer['values'].append(value)
                self.experience_buffer['log_probs'].append(log_prob)
                self.experience_buffer['dones'].append(terminated)
            
            # 7. Atualizar rede neural periodicamente
            if self.training_step_count >= self.ppo_update_freq and TORCH_AVAILABLE and self.neural_network:
                self._update_neural_network()
                self.training_step_count = 0
            
            if self.game_over:
                self._end_training_episode()
                break
            
            # Se em slow motion, sair para renderizar
            if self.slow_motion and self.training_render:
                break
    
    def _get_neural_action(self, obs: np.ndarray):
        """Obtém ação da rede neural"""
        if not TORCH_AVAILABLE or self.neural_network is None:
            # Fallback: ação simples baseada em regras
            return self._get_simple_action(), 0.0, 0.0
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_mean, action_std, value = self.neural_network(obs_tensor)
            
            # Amostrar ação da distribuição
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            return action.squeeze().numpy(), value.item(), log_prob.item()
    
    def _get_simple_action(self) -> np.ndarray:
        """Ação simples de fallback quando não há rede neural"""
        if not self.training_entity or not self.training_opponent:
            return np.zeros(4)
        
        dx = self.training_opponent.x - self.training_entity.x
        dy = self.training_opponent.y - self.training_entity.y
        dist = math.sqrt(dx**2 + dy**2) + 0.001
        
        move_x = dx / dist
        move_y = dy / dist
        attack = 1.0 if dist < 100 else 0.0
        ability = 1.0 if dist < 50 and random.random() > 0.95 else 0.0
        
        return np.array([move_x, move_y, attack, ability])
    
    def _apply_neural_action(self, action: np.ndarray):
        """Aplica ação da rede neural à entidade"""
        if not self.training_entity or not self.training_entity.is_alive():
            return
        
        # Movimento (valores contínuos -1 a 1)
        move_x = float(np.clip(action[0], -1, 1))
        move_y = float(np.clip(action[1], -1, 1))
        
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            self.training_entity.move(move_x, move_y)
        else:
            self.training_entity.moving = False
        
        # Rastrear se tentou atacar neste frame
        self._tried_attack_this_frame = False
        
        # Ataque (threshold 0.5)
        if action[2] > 0.5:
            self._tried_attack_this_frame = True
            self.training_entity.attack()
        
        # Habilidade (threshold 0.5)
        if action[3] > 0.5:
            self.training_entity.use_ability()
    
    def _update_neural_network(self):
        """Atualiza a rede neural usando PPO"""
        if not TORCH_AVAILABLE or self.neural_network is None:
            return
        
        if len(self.experience_buffer['obs']) < self.ppo_batch_size:
            return
        
        # Converter para tensores
        obs_tensor = torch.FloatTensor(np.array(self.experience_buffer['obs']))
        action_tensor = torch.FloatTensor(np.array(self.experience_buffer['actions']))
        old_log_probs = torch.FloatTensor(np.array(self.experience_buffer['log_probs']))
        rewards = np.array(self.experience_buffer['rewards'])
        values = np.array(self.experience_buffer['values'])
        dones = np.array(self.experience_buffer['dones'])
        
        # Calcular vantagens usando GAE
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        # Normalizar vantagens
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        # Múltiplas épocas de atualização
        dataset_size = len(self.experience_buffer['obs'])
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        num_updates = 0
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.ppo_batch_size):
                end = min(start + self.ppo_batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                # Mini-batch
                batch_obs = obs_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # Forward pass
                values_pred, log_probs, entropy = self.neural_network.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                # Ratio para PPO
                ratio = torch.exp(log_probs - batch_old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo_clip_epsilon, 1 + self.ppo_clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values_pred, batch_returns)
                
                # Entropy bonus (encoraja exploração)
                entropy_loss = -entropy.mean()
                
                # Loss total
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # Backward
                self.nn_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), 0.5)
                self.nn_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Atualizar estatísticas
        if num_updates > 0:
            self.training_stats['policy_loss'] = total_policy_loss / num_updates
            self.training_stats['value_loss'] = total_value_loss / num_updates
            self.training_stats['entropy'] = total_entropy / num_updates
            self.training_stats['learning_updates'] += 1
        
        # Limpar buffer
        self._clear_experience_buffer()
        
        print(f"🔄 Atualização #{self.training_stats['learning_updates']}: "
              f"Policy Loss={self.training_stats['policy_loss']:.4f}, "
              f"Value Loss={self.training_stats['value_loss']:.4f}")
    
    def _compute_gae(self, rewards, values, dones):
        """Calcula Generalized Advantage Estimation"""
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0
                gae = 0
            
            delta = rewards[t] + self.ppo_gamma * next_value - values[t]
            gae = delta + self.ppo_gamma * self.ppo_gae_lambda * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        return advantages, returns
    
    def _calculate_training_reward(self) -> float:
        """Calcula a recompensa do passo atual"""
        reward = 0.0
        
        if not hasattr(self, '_prev_agent_health'):
            self._prev_agent_health = self.training_entity.health
        if not hasattr(self, '_prev_opponent_health'):
            self._prev_opponent_health = self.training_opponent.health if self.training_opponent else 0
        if not hasattr(self, '_tried_attack_this_frame'):
            self._tried_attack_this_frame = False
        
        # Calcular dano causado
        damage_dealt = 0
        if self.training_opponent:
            damage_dealt = self._prev_opponent_health - self.training_opponent.health
            if damage_dealt > 0:
                # Recompensa por causar dano
                reward += damage_dealt * 0.1
            self._prev_opponent_health = self.training_opponent.health
        
        # Penalidade por atacar sem causar dano (ataque desperdiçado)
        if self._tried_attack_this_frame and damage_dealt <= 0:
            reward -= 0.02  # Pequena penalidade por ataque no vazio
        
        # Dano recebido
        damage_taken = self._prev_agent_health - self.training_entity.health
        if damage_taken > 0:
            reward -= damage_taken * 0.05
        self._prev_agent_health = self.training_entity.health
        
        # Reset flag de ataque
        self._tried_attack_this_frame = False
        
        return reward
    
    def _end_training_episode(self):
        """Finaliza um episódio de treinamento"""
        # Determinar resultado
        won = self.training_opponent and not self.training_opponent.is_alive()
        
        if won:
            self.training_stats['wins'] += 1
            self.training_episode_reward += 10
        else:
            self.training_stats['losses'] += 1
            self.training_episode_reward -= 10
        
        self.training_stats['episode'] += 1
        self.training_stats['recent_rewards'].append(self.training_episode_reward)
        
        if self.training_episode_reward > self.training_stats['best_reward']:
            self.training_stats['best_reward'] = self.training_episode_reward
        
        if len(self.training_stats['recent_rewards']) > 0:
            self.training_stats['avg_reward'] = sum(self.training_stats['recent_rewards']) / len(self.training_stats['recent_rewards'])
        
        # Verificar se terminou
        if self.training_stats['episode'] >= self.training_stats['total_episodes']:
            self.stop_training()
            return
        
        # Rotacionar oponente se estiver no modo multi-oponente
        if self.training_multi_opponent:
            if self.training_stats['episode'] % self.episodes_per_opponent == 0:
                self._rotate_opponent()
        
        # Resetar para próximo episódio
        agent_class = self.available_classes[self.training_class]
        agent_weapon = self.available_weapons[self.training_weapon]
        
        # Obter oponente atual (pode ser rotacionado)
        opponent_class, opponent_weapon = self._get_current_opponent_config()
        
        self._prev_agent_health = 100
        self._prev_opponent_health = 100
        
        self.reset_training_episode(agent_class, agent_weapon, opponent_class, opponent_weapon)
    
    def stop_training(self):
        """Para o treinamento"""
        self.training_running = False
        elapsed = time.time() - self.training_stats['training_time']
        self.training_stats['training_time'] = elapsed
    
    def save_current_model(self):
        """Salva o modelo atual (pesos da rede neural + estatísticas)"""
        agent_class = self.available_classes[self.training_class]
        agent_weapon = self.available_weapons[self.training_weapon]
        
        if self.training_multi_opponent:
            opponent_class = "multi"
            opponent_weapon = "multi"
        else:
            opponent_class = self.available_classes[self.training_opponent_class]
            opponent_weapon = self.available_weapons[self.training_opponent_weapon]
        
        # Criar pasta se não existir
        os.makedirs("models", exist_ok=True)
        
        # Nome base do arquivo
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        opp_suffix = "multi" if self.training_multi_opponent else f"{opponent_class}_{opponent_weapon}"
        base_name = f"model_{agent_class}_{agent_weapon}_vs_{opp_suffix}_ep{self.training_stats['episode']}_{timestamp}"
        
        # Salvar pesos da rede neural se disponível
        has_neural_weights = False
        if TORCH_AVAILABLE and self.neural_network is not None:
            weights_path = f"models/{base_name}.pth"
            torch.save({
                'model_state_dict': self.neural_network.state_dict(),
                'optimizer_state_dict': self.nn_optimizer.state_dict(),
                'obs_size': self.nn_obs_size,
                'action_size': self.nn_action_size
            }, weights_path)
            has_neural_weights = True
            print(f"🧠 Pesos da rede neural salvos em: {weights_path}")
        
        # Salvar estatísticas do treinamento
        model_data = {
            'agent_class': agent_class,
            'agent_weapon': agent_weapon,
            'opponent_class': opponent_class,
            'opponent_weapon': opponent_weapon,
            'multi_opponent': self.training_multi_opponent,
            'episode': self.training_stats['episode'],
            'total_episodes': self.training_stats['total_episodes'],
            'wins': self.training_stats['wins'],
            'losses': self.training_stats['losses'],
            'win_rate': self.training_stats['wins'] / max(1, self.training_stats['episode']) * 100,
            'avg_reward': self.training_stats['avg_reward'],
            'best_reward': self.training_stats['best_reward'],
            'training_time': time.time() - self.training_stats['training_time'] if isinstance(self.training_stats['training_time'], float) and self.training_stats['training_time'] > 1000000 else self.training_stats['training_time'],
            'has_neural_weights': has_neural_weights,
            'weights_file': f"{base_name}.pth" if has_neural_weights else None,
            'policy_loss': self.training_stats.get('policy_loss', 0),
            'value_loss': self.training_stats.get('value_loss', 0),
            'learning_updates': self.training_stats.get('learning_updates', 0)
        }
        
        json_path = f"models/{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        # Feedback visual
        self.save_message = f"✅ Modelo salvo: {base_name}"
        self.save_message_timer = 3.0
        
        print(f"✅ Estatísticas salvas em: {json_path}")
    
    def _load_game_ai_models(self):
        """Carrega lista de modelos de IA disponíveis para gameplay"""
        import json
        
        self.game_ai_models = []
        self.selected_ai_model_idx = 0
        
        if os.path.exists("models"):
            files = [f for f in os.listdir("models") if f.endswith('.json')]
            for f in files:
                try:
                    with open(os.path.join("models", f), 'r') as file:
                        data = json.load(file)
                        data['filename'] = f
                        # Só adiciona se tiver pesos neurais
                        if data.get('has_neural_weights'):
                            self.game_ai_models.append(data)
                except:
                    pass
    
    def _open_tournament_ui(self):
        """Abre a interface visual do sistema de torneio em grupo"""
        global screen, SCREEN_WIDTH, SCREEN_HEIGHT, font, font_large, font_title, font_sizes
        
        # Salvar tamanho atual
        saved_width = SCREEN_WIDTH
        saved_height = SCREEN_HEIGHT
        
        # Fechar pygame atual
        pygame.quit()
        
        # Importar e executar a UI do torneio
        try:
            from tournament_ui import TournamentUI
            
            # Obter tamanho do monitor
            pygame.init()
            info = pygame.display.Info()
            width = int(info.current_w * 0.85)
            height = int(info.current_h * 0.85)
            pygame.quit()
            
            # Executar UI do torneio
            ui = TournamentUI(width, height)
            ui.run()
            
        except Exception as e:
            print(f"Erro ao abrir torneio: {e}")
        
        # Restaurar janela do jogo principal completamente
        pygame.init()
        
        # Restaurar tamanho da tela
        SCREEN_WIDTH = saved_width
        SCREEN_HEIGHT = saved_height
        
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("Circle Warriors - Weapon Ball Fight")
        
        # Restaurar fontes
        font_sizes = get_font_sizes()
        font = pygame.font.Font(None, font_sizes['small'])
        font_large = pygame.font.Font(None, font_sizes['medium'])
        font_title = pygame.font.Font(None, font_sizes['large'])
        
        # Restaurar arena e física
        self.arena_config = ArenaConfig(SCREEN_WIDTH, SCREEN_HEIGHT, 50)
        self.physics = Physics(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.map_renderer = MapRenderer(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    def _apply_selected_ai_model(self):
        """Carrega o modelo de IA selecionado para usar no gameplay"""
        self.loaded_game_ai = None
        self.use_trained_ai = False
        
        # Índice 0 = IA Simples (sem modelo treinado)
        if self.selected_ai_model_idx == 0 or not self.game_ai_models:
            return
        
        # Carregar modelo selecionado (índice - 1 porque 0 é "IA Simples")
        model = self.game_ai_models[self.selected_ai_model_idx - 1]
        
        if model.get('has_neural_weights') and TORCH_AVAILABLE:
            weights_file = model.get('weights_file')
            if weights_file:
                weights_path = os.path.join("models", weights_file)
                if os.path.exists(weights_path):
                    try:
                        checkpoint = torch.load(weights_path)
                        
                        obs_size = checkpoint.get('obs_size', self.nn_obs_size)
                        action_size = checkpoint.get('action_size', self.nn_action_size)
                        
                        self.loaded_game_ai = ActorCritic(
                            obs_size=obs_size,
                            action_size=action_size,
                            hidden_size=256
                        )
                        self.loaded_game_ai.load_state_dict(checkpoint['model_state_dict'])
                        self.loaded_game_ai.eval()
                        
                        self.use_trained_ai = True
                        print(f"🧠 IA Neural carregada para gameplay: {weights_file}")
                    except Exception as e:
                        print(f"⚠️ Erro ao carregar IA: {e}")
                        self.loaded_game_ai = None
    
    def load_trained_model(self):
        """Abre o menu de carregar modelos"""
        import json
        
        self.available_models = []
        self.selected_model_idx = 0
        
        if os.path.exists("models"):
            files = [f for f in os.listdir("models") if f.endswith('.json')]
            for f in files:
                try:
                    with open(os.path.join("models", f), 'r') as file:
                        data = json.load(file)
                        data['filename'] = f
                        self.available_models.append(data)
                except:
                    pass
        
        self.load_menu_active = True
    
    def load_selected_model(self):
        """Carrega o modelo selecionado e inicia visualização com a rede neural treinada"""
        if not self.available_models:
            return
        
        model = self.available_models[self.selected_model_idx]
        
        # Configurar classe/arma baseado no modelo
        agent_class = model.get('agent_class', 'warrior')
        agent_weapon = model.get('agent_weapon', 'sword')
        opponent_class = model.get('opponent_class', 'warrior')
        opponent_weapon = model.get('opponent_weapon', 'sword')
        
        # Encontrar índices
        if agent_class in self.available_classes:
            self.training_class = self.available_classes.index(agent_class)
        if agent_weapon in self.available_weapons:
            self.training_weapon = self.available_weapons.index(agent_weapon)
        if opponent_class in self.available_classes:
            self.training_opponent_class = self.available_classes.index(opponent_class)
        if opponent_weapon in self.available_weapons:
            self.training_opponent_weapon = self.available_weapons.index(opponent_weapon)
        
        # Carregar pesos da rede neural se disponível
        neural_loaded = False
        if model.get('has_neural_weights') and TORCH_AVAILABLE:
            weights_file = model.get('weights_file')
            if weights_file:
                weights_path = os.path.join("models", weights_file)
                if os.path.exists(weights_path):
                    try:
                        checkpoint = torch.load(weights_path)
                        
                        # Criar a rede neural com as dimensões corretas
                        obs_size = checkpoint.get('obs_size', self.nn_obs_size)
                        action_size = checkpoint.get('action_size', self.nn_action_size)
                        
                        self.neural_network = ActorCritic(
                            obs_size=obs_size,
                            action_size=action_size,
                            hidden_size=256
                        )
                        self.neural_network.load_state_dict(checkpoint['model_state_dict'])
                        self.neural_network.eval()  # Modo avaliação (não treinamento)
                        
                        # Recriar otimizador
                        self.nn_optimizer = torch.optim.Adam(
                            self.neural_network.parameters(), 
                            lr=3e-4
                        )
                        self.nn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        
                        neural_loaded = True
                        print(f"🧠 Rede neural carregada de: {weights_path}")
                    except Exception as e:
                        print(f"⚠️ Erro ao carregar pesos: {e}")
                        self.neural_network = None
        
        # Carregar stats
        self.training_stats = {
            'episode': model.get('episode', 0),
            'total_episodes': model.get('total_episodes', 0),
            'wins': model.get('wins', 0),
            'losses': model.get('losses', 0),
            'avg_reward': model.get('avg_reward', 0),
            'recent_rewards': deque(maxlen=100),
            'best_reward': model.get('best_reward', 0),
            'training_time': model.get('training_time', 0),
            'policy_loss': model.get('policy_loss', 0),
            'value_loss': model.get('value_loss', 0),
            'learning_updates': model.get('learning_updates', 0)
        }
        
        # Iniciar modo de teste/visualização
        self.load_menu_active = False
        self.training_running = True
        self.training_paused = True  # Começa pausado para ver
        self.training_render = True
        self.mode = "training"
        
        # Limpar buffer de experiência
        self._clear_experience_buffer()
        
        # Iniciar episódio
        self.reset_training_episode(agent_class, agent_weapon, opponent_class, opponent_weapon)
        
        # Feedback
        if neural_loaded:
            self.save_message = f"🧠 IA Neural carregada: {model['filename']}"
        else:
            self.save_message = f"⚠️ Modelo sem rede neural: {model['filename']}"
        self.save_message_timer = 3.0
    
    def test_trained_model(self):
        """Testa um modelo treinado"""
        # Mesmo que carregar, mas inicia rodando
        self.load_trained_model()

    def update(self, dt: float):
        """Atualiza o estado do jogo"""
        if self.mode == "training":
            self.update_training(dt)
            return
        
        if self.mode == "group_battle":
            self._update_group_battle(dt)
            return
        
        if self.paused or self.game_over or self.mode in ["menu", "select", "training_menu", "training_select", "group_menu"]:
            return
        
        # Atualizar slow motion timer
        if self.slow_motion_timer > 0:
            self.slow_motion_timer -= dt
            if self.slow_motion_timer <= 0:
                self.slow_motion = False
        
        # Aplicar slow motion ao dt
        effective_dt = dt * self.slow_motion_scale if self.slow_motion else dt
        
        # Guardar vida anterior das entidades para detectar dano
        health_before = {id(e): e.health for e in self.entities if e.is_alive()}
        
        # Atualizar entidades
        for entity in self.entities:
            if entity.is_alive():
                entity.update(effective_dt)
        
        # Verificar armadilhas do Trapper
        for entity in self.entities:
            if hasattr(entity, 'check_traps') and entity.is_alive():
                # Inimigos são: entidades diferentes, vivas, e de outro team (ou simplesmente diferentes se sem team)
                enemies = [e for e in self.entities if e != entity and e.is_alive() and (entity.team == "none" or e.team != entity.team)]
                entity.check_traps(enemies)
        
        # Verificar armadilhas do TrapLauncher (arma)
        for entity in self.entities:
            if entity.is_alive() and hasattr(entity, 'weapon') and entity.weapon:
                if hasattr(entity.weapon, 'check_trap_hits'):
                    hits = entity.weapon.check_trap_hits(self.entities)
                    for target, trap in hits:
                        # Aplicar dano e root
                        target.take_damage(trap['damage'], entity)
                        from stats import StatusEffect, StatusEffectType
                        target.apply_status_effect(StatusEffect(
                            name='launcher_trap_root',
                            effect_type=StatusEffectType.ROOT,
                            duration=trap['root_duration'],
                            source=entity
                        ))
        
        # Verificar armas especiais (Staff cura, Tome buffa)
        self._check_special_weapons()
        
        # Física
        self.physics.handle_collisions(self.entities)
        
        # Sistema de mapa grande
        if self.large_map_enabled and self.obstacle_manager:
            # Usar limites do mapa grande
            large_arena_rect = pygame.Rect(0, 0, self.large_map_width, self.large_map_height)
            
            # Colisão com obstáculos
            for entity in self.entities:
                if entity.is_alive():
                    self.obstacle_manager.resolve_collision(entity, effective_dt)
                    self.physics.constrain_to_arena(entity, large_arena_rect)
            
            # Colisão de projéteis com bordas e obstáculos
            self.physics.check_projectiles_arena_collision(self.entities, large_arena_rect)
            self._check_projectiles_obstacles()
            
            # Atualizar câmera
            if self.camera and self.camera_target:
                self.camera.set_target(self.camera_target)
                self.camera.update(effective_dt)
        else:
            # Limitar à arena normal
            arena_rect = pygame.Rect(*self.arena_config.playable_rect)
            for entity in self.entities:
                self.physics.constrain_to_arena(entity, arena_rect)
            
            # Colisão de projéteis com bordas da arena
            self.physics.check_projectiles_arena_collision(self.entities, arena_rect)
        
        # Detectar se alguém tomou dano para ativar slow motion
        for entity in self.entities:
            entity_id = id(entity)
            if entity_id in health_before:
                if entity.health < health_before[entity_id]:
                    # Alguém tomou dano! Ativar slow motion
                    self.slow_motion = True
                    self.slow_motion_timer = self.slow_motion_duration
                    break
        
        # Avançar estado do jogo
        self.game_state.step()
        
        # Verificar fim de jogo
        alive = [e for e in self.entities if e.is_alive()]
        if len(alive) <= 1:
            self.game_over = True
            self.winner = alive[0] if alive else None
    
    def draw(self):
        """Desenha o jogo"""
        screen.fill(DARK_GRAY)
        
        if self.mode == "menu":
            self.draw_menu()
        elif self.mode == "editor":
            self.draw_editor()
        elif self.mode == "select":
            self.draw_selection()
        elif self.mode == "group_menu":
            self.draw_group_menu()
        elif self.mode == "group_battle":
            self.draw_group_battle()
        elif self.mode == "training_menu":
            if self.model_stats_view_active:
                self.draw_model_stats_view()
            elif self.load_menu_active:
                self.draw_load_menu()
            else:
                self.draw_training_menu()
        elif self.mode == "training_select":
            self.draw_training_selection()
        elif self.mode == "training":
            self.draw_training()
        else:
            self.draw_game()
        
        pygame.display.flip()
    
    def draw_menu(self):
        """Desenha o menu principal"""
        # Título
        title = font_title.render("CIRCLE WARRIORS", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
        screen.blit(title, title_rect)
        
        subtitle = font_large.render("Weapon Ball Fight", True, GRAY)
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 120))
        screen.blit(subtitle, sub_rect)
        
        # Opções
        options = [
            ("1 - Jogador vs Jogador (PvP)", 180),
            ("2 - Jogador vs IA (PvE)", 220),
            ("3 - IA vs IA (Assistir)", 260),
            ("4 - Treinar IA 🧠", 310),
            ("5 - Batalha em Grupo ⚔️", 350),
            ("6 - Editor de Atributos ⚙", 390),
        ]
        
        for text, y in options:
            if "Treinar" in text:
                color = (100, 255, 150)
            elif "Batalha em Grupo" in text:
                color = (255, 150, 255)
            elif "Editor" in text:
                color = (255, 200, 100)
            else:
                color = WHITE
            surf = font_large.render(text, True, color)
            rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
            screen.blit(surf, rect)
        
        # Divisor
        pygame.draw.line(screen, GRAY, (100, 440), (SCREEN_WIDTH - 100, 440), 2)
        
        # Lista de classes
        class_title = font_large.render("Classes Disponíveis:", True, (100, 200, 255))
        screen.blit(class_title, (100, 420))
        
        classes = ClassRegistry.get_all()
        class_y = 460
        for i, (class_id, cls) in enumerate(classes.items()):
            color = (200, 200, 200) if i % 2 == 0 else (180, 180, 180)
            text = f"• {cls.display_name}: {cls.description}"
            surf = font.render(text, True, color)
            screen.blit(surf, (120, class_y + i * 22))
        
        # Lista de armas
        weapons_y = class_y + len(classes) * 22 + 10
        weapons_title = font_large.render("Armas Disponíveis:", True, (255, 200, 100))
        screen.blit(weapons_title, (100, weapons_y))
        
        weapons = WeaponRegistry.get_all()
        for i, (weapon_id, weapon_cls) in enumerate(weapons.items()):
            color = (200, 200, 200) if i % 2 == 0 else (180, 180, 180)
            text = f"• {weapon_cls.display_name}: {weapon_cls.description}"
            surf = font.render(text, True, color)
            screen.blit(surf, (120, weapons_y + 28 + i * 22))
        
        # Instruções
        instructions = font.render("ESC - Sair", True, GRAY)
        screen.blit(instructions, (50, SCREEN_HEIGHT - 40))
    
    def draw_editor(self):
        """Desenha o editor de atributos de classes e armas"""
        # Título
        title = font_title.render("EDITOR DE ATRIBUTOS", True, (255, 200, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title, title_rect)
        
        # Subtítulo mostrando o que está editando
        mode_text = "CLASSES" if self.editor_mode == "classes" else "ARMAS"
        mode_color = (100, 200, 255) if self.editor_mode == "classes" else (255, 150, 100)
        subtitle = font_large.render(f"Editando: {mode_text} (TAB para alternar)", True, mode_color)
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 95))
        screen.blit(subtitle, sub_rect)
        
        # Item atual
        items = self.available_classes if self.editor_mode == "classes" else self.available_weapons
        item_id = items[self.editor_selected_item]
        
        # Obter nome do display
        if self.editor_mode == "classes":
            cls = ClassRegistry.get_all()[item_id]
            item_name = cls.display_name
            item_desc = cls.description
        else:
            weapon_cls = WeaponRegistry.get_all()[item_id]
            item_name = weapon_cls.display_name
            item_desc = weapon_cls.description
        
        # Caixa do item selecionado
        box_x = SCREEN_WIDTH // 2 - 250
        box_y = 130
        box_w = 500
        box_h = 80
        
        # Fundo semi-transparente
        s = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        s.fill((40, 40, 60, 200))
        screen.blit(s, (box_x, box_y))
        pygame.draw.rect(screen, mode_color, (box_x, box_y, box_w, box_h), 3)
        
        # Setas de navegação
        arrow_left = font_title.render("◄", True, (200, 200, 200))
        arrow_right = font_title.render("►", True, (200, 200, 200))
        screen.blit(arrow_left, (box_x - 50, box_y + 20))
        screen.blit(arrow_right, (box_x + box_w + 15, box_y + 20))
        
        # Nome do item
        name_surf = font_large.render(f"{self.editor_selected_item + 1}/{len(items)}: {item_name}", True, WHITE)
        name_rect = name_surf.get_rect(center=(box_x + box_w // 2, box_y + 30))
        screen.blit(name_surf, name_rect)
        
        # Descrição
        desc_surf = font.render(item_desc, True, GRAY)
        desc_rect = desc_surf.get_rect(center=(box_x + box_w // 2, box_y + 55))
        screen.blit(desc_surf, desc_rect)
        
        # Stats do item
        stats = self._get_current_editor_stats()
        stats_y = 230
        
        # Títulos das colunas
        stat_title = font_large.render("ATRIBUTO", True, (150, 150, 150))
        val_title = font_large.render("VALOR", True, (150, 150, 150))
        screen.blit(stat_title, (SCREEN_WIDTH // 2 - 200, stats_y))
        screen.blit(val_title, (SCREEN_WIDTH // 2 + 80, stats_y))
        
        # Linha divisória
        pygame.draw.line(screen, (100, 100, 100), 
                        (SCREEN_WIDTH // 2 - 230, stats_y + 30), 
                        (SCREEN_WIDTH // 2 + 230, stats_y + 30), 2)
        
        # Nomes amigáveis dos stats
        stat_names = {
            'max_health': 'Vida Máxima',
            'speed': 'Velocidade',
            'acceleration': 'Aceleração',
            'defense': 'Defesa',
            'damage_multiplier': 'Multi. Dano',
            'base_damage': 'Dano Base',
            'range': 'Alcance',
            'attack_cooldown': 'Cooldown Ataque',
            'knockback_force': 'Força Knockback',
            'critical_chance': 'Chance Crítico'
        }
        
        for i, (stat_key, stat_value) in enumerate(stats):
            y = stats_y + 50 + i * 45
            
            # Destacar o selecionado
            if i == self.editor_selected_stat:
                # Fundo de seleção
                sel_surf = pygame.Surface((500, 40), pygame.SRCALPHA)
                sel_surf.fill((100, 100, 200, 100))
                screen.blit(sel_surf, (SCREEN_WIDTH // 2 - 240, y - 5))
                
                # Indicador de seleção
                indicator = font_large.render("►", True, (255, 255, 100))
                screen.blit(indicator, (SCREEN_WIDTH // 2 - 260, y))
            
            # Nome do stat
            display_name = stat_names.get(stat_key, stat_key)
            name_color = (255, 255, 100) if i == self.editor_selected_stat else (200, 200, 200)
            stat_surf = font_large.render(display_name, True, name_color)
            screen.blit(stat_surf, (SCREEN_WIDTH // 2 - 200, y))
            
            # Valor (formatado)
            if isinstance(stat_value, float):
                if stat_key == 'critical_chance':
                    val_text = f"{stat_value * 100:.0f}%"
                elif stat_value < 1:
                    val_text = f"{stat_value:.2f}"
                else:
                    val_text = f"{stat_value:.1f}"
            else:
                val_text = str(stat_value)
            
            val_color = (100, 255, 150) if i == self.editor_selected_stat else (150, 200, 150)
            val_surf = font_large.render(val_text, True, val_color)
            screen.blit(val_surf, (SCREEN_WIDTH // 2 + 100, y))
            
            # Botões +/- visuais para o selecionado
            if i == self.editor_selected_stat:
                minus_surf = font_large.render("[ - ]", True, (255, 100, 100))
                plus_surf = font_large.render("[ + ]", True, (100, 255, 100))
                screen.blit(minus_surf, (SCREEN_WIDTH // 2 + 160, y))
                screen.blit(plus_surf, (SCREEN_WIDTH // 2 + 210, y))
        
        # Instruções
        instr_y = SCREEN_HEIGHT - 100
        
        instructions = [
            "◄/► ou A/D: Trocar Item",
            "▲/▼ ou W/S: Selecionar Atributo",
            "+/-: Ajustar Valor",
            "R: Resetar para Padrão",
            "TAB: Alternar Classes/Armas",
            "ESC: Voltar ao Menu"
        ]
        
        # Desenhar instruções em duas colunas
        instr_box = pygame.Surface((SCREEN_WIDTH - 100, 80), pygame.SRCALPHA)
        instr_box.fill((30, 30, 50, 180))
        screen.blit(instr_box, (50, instr_y - 10))
        
        for i, text in enumerate(instructions):
            col = i % 3
            row = i // 3
            x = 80 + col * 270
            y = instr_y + row * 25
            color = (180, 180, 200)
            surf = font.render(text, True, color)
            screen.blit(surf, (x, y))
    
    def draw_selection(self):
        """Desenha a tela de seleção de personagem"""
        player = self.current_player_selecting
        player_name = "Jogador 1" if player == 1 else "Jogador 2"
        
        if self.game_mode_choice == "pve" and player == 2:
            player_name = "IA Inimiga"
        elif self.game_mode_choice == "ai_vs_ai":
            player_name = f"IA {player}"
        
        # Cor do jogador
        player_color = (255, 100, 100) if player == 1 else (100, 100, 255)
        
        # Título
        if self.selection_step in ["map", "ai_model"]:
            title_text = "Escolha o MAPA" if self.selection_step == "map" else "Escolha a IA"
            title = font_title.render(title_text, True, (100, 255, 200))
        else:
            title = font_title.render(f"Seleção - {player_name}", True, player_color)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 60))
        screen.blit(title, title_rect)
        
        # Subtítulo (o que está selecionando)
        if self.selection_step == "class":
            step_text = "Escolha a CLASSE"
        elif self.selection_step == "weapon":
            step_text = "Escolha a ARMA"
        elif self.selection_step == "map":
            step_text = "← →  para navegar entre os mapas"
        elif self.selection_step == "ai_model":
            step_text = "Selecione o modelo de IA treinada"
        else:
            step_text = ""
        
        step_surf = font_large.render(step_text, True, WHITE)
        step_rect = step_surf.get_rect(center=(SCREEN_WIDTH // 2, 110))
        screen.blit(step_surf, step_rect)
        
        # Área de seleção
        if self.selection_step == "class":
            self.draw_class_selection(player)
        elif self.selection_step == "weapon":
            self.draw_weapon_selection(player)
        elif self.selection_step == "map":
            self.draw_map_selection()
        elif self.selection_step == "ai_model":
            self.draw_ai_model_selection()
        
        # Seleção atual do jogador 1 (se estiver no jogador 2 e em class/weapon)
        if player == 2 and self.selection_step in ["class", "weapon"]:
            p1_class = self.available_classes[self.selected_class_p1]
            p1_weapon = self.available_weapons[self.selected_weapon_p1]
            p1_class_name = ClassRegistry.get_all()[p1_class].display_name
            p1_weapon_name = WeaponRegistry.get_all()[p1_weapon].display_name
            
            p1_info = font.render(f"P1: {p1_class_name} + {p1_weapon_name}", True, (255, 100, 100))
            screen.blit(p1_info, (50, SCREEN_HEIGHT - 80))
        
        # Instruções
        instructions = [
            "← → ou A/D - Navegar",
            "ENTER/ESPAÇO - Confirmar",
            "BACKSPACE - Voltar",
            "ESC - Menu"
        ]
        for i, text in enumerate(instructions):
            surf = font.render(text, True, GRAY)
            screen.blit(surf, (SCREEN_WIDTH - 220, SCREEN_HEIGHT - 120 + i * 25))
    
    def draw_class_selection(self, player: int):
        """Desenha a seleção de classes"""
        classes = ClassRegistry.get_all()
        class_list = list(classes.items())
        selected_idx = self.selected_class_p1 if player == 1 else self.selected_class_p2
        
        # Calcular posições
        total_width = len(class_list) * 180
        start_x = (SCREEN_WIDTH - total_width) // 2 + 90
        
        for i, (class_id, cls) in enumerate(class_list):
            x = start_x + i * 180
            y = 280
            
            # Determinar se está selecionado
            is_selected = (i == selected_idx)
            
            # Cor do círculo
            if is_selected:
                circle_color = (255, 100, 100) if player == 1 else (100, 100, 255)
                border_color = WHITE
                border_width = 4
            else:
                circle_color = (80, 80, 80)
                border_color = GRAY
                border_width = 2
            
            # Desenhar círculo representando a classe
            radius = 50 if is_selected else 40
            pygame.draw.circle(screen, circle_color, (x, y), radius)
            pygame.draw.circle(screen, border_color, (x, y), radius, border_width)
            
            # Nome da classe
            name_color = WHITE if is_selected else GRAY
            name_surf = font_large.render(cls.display_name, True, name_color)
            name_rect = name_surf.get_rect(center=(x, y + radius + 30))
            screen.blit(name_surf, name_rect)
            
            # Indicador de seleção
            if is_selected:
                # Setas
                arrow_y = y - radius - 20
                pygame.draw.polygon(screen, WHITE, [
                    (x, arrow_y - 15),
                    (x - 10, arrow_y),
                    (x + 10, arrow_y)
                ])
        
        # Descrição da classe selecionada
        selected_class_id = class_list[selected_idx][0]
        selected_class = class_list[selected_idx][1]
        
        # Box de descrição
        desc_box = pygame.Rect(100, 450, SCREEN_WIDTH - 200, 150)
        pygame.draw.rect(screen, (40, 40, 40), desc_box)
        pygame.draw.rect(screen, GRAY, desc_box, 2)
        
        # Buscar stats do banco de dados central
        custom_stats = self.config_db.get_class_stats(selected_class_id)
        
        desc_title = font_large.render(f"{selected_class.display_name}", True, WHITE)
        screen.blit(desc_title, (120, 460))
        
        desc_text = font.render(selected_class.description, True, (200, 200, 200))
        screen.blit(desc_text, (120, 495))
        
        # Stats visuais (usando valores do banco de dados)
        stats_info = [
            ("Vida", custom_stats.get('max_health', 100), 150, (100, 255, 100)),
            ("Velocidade", custom_stats.get('speed', 2.5), 10, (100, 200, 255)),
            ("Dano", custom_stats.get('damage_multiplier', 1.0), 2, (255, 100, 100)),
            ("Defesa", custom_stats.get('defense', 0), 30, (200, 200, 100)),
        ]
        
        stat_x = 120
        for stat_name, value, max_val, color in stats_info:
            stat_y = 530
            
            # Nome
            name_surf = font.render(stat_name, True, WHITE)
            screen.blit(name_surf, (stat_x, stat_y))
            
            # Barra
            bar_width = 80
            bar_height = 12
            fill_width = min(bar_width, (value / max_val) * bar_width)
            
            pygame.draw.rect(screen, (60, 60, 60), (stat_x, stat_y + 22, bar_width, bar_height))
            pygame.draw.rect(screen, color, (stat_x, stat_y + 22, fill_width, bar_height))
            pygame.draw.rect(screen, WHITE, (stat_x, stat_y + 22, bar_width, bar_height), 1)
            
            stat_x += 140
        
        # Habilidade
        ability_info = selected_class(0, 0, (255, 255, 255)).get_ability_info()
        ability_text = font.render(f"Habilidade: {ability_info['name']} - {ability_info['description']}", True, (180, 180, 255))
        screen.blit(ability_text, (120, 570))
    
    def draw_weapon_selection(self, player: int):
        """Desenha a seleção de armas"""
        weapons = WeaponRegistry.get_all()
        weapon_list = list(weapons.items())
        selected_idx = self.selected_weapon_p1 if player == 1 else self.selected_weapon_p2
        
        # Calcular posições
        total_width = len(weapon_list) * 200
        start_x = (SCREEN_WIDTH - total_width) // 2 + 100
        
        for i, (weapon_id, weapon_cls) in enumerate(weapon_list):
            x = start_x + i * 200
            y = 280
            
            is_selected = (i == selected_idx)
            
            # Box da arma
            box_size = 100 if is_selected else 80
            box_rect = pygame.Rect(x - box_size//2, y - box_size//2, box_size, box_size)
            
            if is_selected:
                pygame.draw.rect(screen, (60, 60, 80), box_rect)
                pygame.draw.rect(screen, WHITE, box_rect, 3)
            else:
                pygame.draw.rect(screen, (40, 40, 50), box_rect)
                pygame.draw.rect(screen, GRAY, box_rect, 2)
            
            # Ícone simples da arma (linha representando)
            weapon_color = (255, 255, 200) if is_selected else (150, 150, 150)
            if weapon_id == "sword":
                pygame.draw.line(screen, weapon_color, (x - 20, y + 20), (x + 20, y - 20), 4)
            elif weapon_id == "greatsword":
                pygame.draw.line(screen, weapon_color, (x - 30, y + 30), (x + 30, y - 30), 6)
            elif weapon_id == "dagger":
                pygame.draw.line(screen, weapon_color, (x - 10, y + 10), (x + 10, y - 10), 3)
            elif weapon_id == "spear":
                pygame.draw.line(screen, weapon_color, (x - 35, y), (x + 35, y), 3)
                pygame.draw.polygon(screen, weapon_color, [(x + 35, y), (x + 25, y - 8), (x + 25, y + 8)])
            
            # Nome
            name_color = WHITE if is_selected else GRAY
            name_surf = font_large.render(weapon_cls.display_name, True, name_color)
            name_rect = name_surf.get_rect(center=(x, y + box_size//2 + 30))
            screen.blit(name_surf, name_rect)
            
            # Indicador
            if is_selected:
                arrow_y = y - box_size//2 - 20
                pygame.draw.polygon(screen, WHITE, [
                    (x, arrow_y - 15),
                    (x - 10, arrow_y),
                    (x + 10, arrow_y)
                ])
        
        # Descrição da arma selecionada
        selected_weapon_id = weapon_list[selected_idx][0]
        selected_weapon_cls = weapon_list[selected_idx][1]
        
        # Buscar stats do banco de dados central
        custom_stats = self.config_db.get_weapon_stats(selected_weapon_id)
        
        # Box de descrição
        desc_box = pygame.Rect(100, 450, SCREEN_WIDTH - 200, 150)
        pygame.draw.rect(screen, (40, 40, 40), desc_box)
        pygame.draw.rect(screen, GRAY, desc_box, 2)
        
        desc_title = font_large.render(f"{selected_weapon_cls.display_name}", True, WHITE)
        screen.blit(desc_title, (120, 460))
        
        desc_text = font.render(selected_weapon_cls.description, True, (200, 200, 200))
        screen.blit(desc_text, (120, 495))
        
        # Stats da arma (usando valores do banco de dados)
        stats_info = [
            ("Dano", custom_stats.get('base_damage', 20), 40, (255, 100, 100)),
            ("Velocidade", 1/custom_stats.get('attack_cooldown', 0.5), 5, (100, 200, 255)),
            ("Alcance", custom_stats.get('range', 60), 100, (100, 255, 100)),
            ("Knockback", custom_stats.get('knockback_force', 10), 30, (255, 200, 100)),
            ("Crítico", custom_stats.get('critical_chance', 0.1) * 100, 50, (255, 100, 255)),
        ]
        
        stat_x = 120
        for stat_name, value, max_val, color in stats_info:
            stat_y = 530
            
            name_surf = font.render(stat_name, True, WHITE)
            screen.blit(name_surf, (stat_x, stat_y))
            
            bar_width = 70
            bar_height = 12
            fill_width = min(bar_width, (value / max_val) * bar_width)
            
            pygame.draw.rect(screen, (60, 60, 60), (stat_x, stat_y + 22, bar_width, bar_height))
            pygame.draw.rect(screen, color, (stat_x, stat_y + 22, fill_width, bar_height))
            pygame.draw.rect(screen, WHITE, (stat_x, stat_y + 22, bar_width, bar_height), 1)
            
            stat_x += 120
    
    def draw_map_selection(self):
        """Desenha a seleção de mapas"""
        maps = MapRegistry.get_all()
        map_list = list(maps.items())
        selected_idx = self.selected_map
        
        # Preview do mapa selecionado no centro
        selected_map_id = map_list[selected_idx][0]
        selected_map = map_list[selected_idx][1]
        
        # Área do preview
        preview_width = int(SCREEN_WIDTH * 0.6)
        preview_height = int(SCREEN_HEIGHT * 0.35)
        preview_x = (SCREEN_WIDTH - preview_width) // 2
        preview_y = 140
        
        # Desenhar preview do mapa (mini arena)
        preview_rect = pygame.Rect(preview_x, preview_y, preview_width, preview_height)
        
        # Fundo do mapa
        pygame.draw.rect(screen, selected_map.bg_color, preview_rect)
        
        # Mini arena
        mini_margin = 30
        arena_rect = pygame.Rect(
            preview_x + mini_margin, 
            preview_y + mini_margin,
            preview_width - 2 * mini_margin,
            preview_height - 2 * mini_margin
        )
        pygame.draw.rect(screen, selected_map.floor_color, arena_rect)
        pygame.draw.rect(screen, selected_map.border_color, arena_rect, 3)
        
        # Detalhes decorativos
        center_x = preview_x + preview_width // 2
        center_y = preview_y + preview_height // 2
        pygame.draw.circle(screen, selected_map.detail_color, (center_x, center_y), 40, 2)
        pygame.draw.line(screen, selected_map.detail_color,
                        (center_x, preview_y + mini_margin),
                        (center_x, preview_y + preview_height - mini_margin), 2)
        
        # Borda do preview
        pygame.draw.rect(screen, WHITE, preview_rect, 3)
        
        # Nome do mapa
        name_surf = font_title.render(selected_map.display_name, True, WHITE)
        name_rect = name_surf.get_rect(center=(SCREEN_WIDTH // 2, preview_y + preview_height + 30))
        screen.blit(name_surf, name_rect)
        
        # Descrição
        desc_surf = font.render(selected_map.description, True, (200, 200, 200))
        desc_rect = desc_surf.get_rect(center=(SCREEN_WIDTH // 2, preview_y + preview_height + 60))
        screen.blit(desc_surf, desc_rect)
        
        # Indicadores de navegação
        arrow_y = preview_y + preview_height // 2
        
        # Seta esquerda
        pygame.draw.polygon(screen, WHITE, [
            (preview_x - 40, arrow_y),
            (preview_x - 20, arrow_y - 20),
            (preview_x - 20, arrow_y + 20)
        ])
        
        # Seta direita
        pygame.draw.polygon(screen, WHITE, [
            (preview_x + preview_width + 40, arrow_y),
            (preview_x + preview_width + 20, arrow_y - 20),
            (preview_x + preview_width + 20, arrow_y + 20)
        ])
        
        # Indicador de posição (bolinhas)
        dots_y = preview_y + preview_height + 90
        dot_spacing = 20
        total_dots_width = len(map_list) * dot_spacing
        dots_start_x = (SCREEN_WIDTH - total_dots_width) // 2
        
        for i, _ in enumerate(map_list):
            dot_x = dots_start_x + i * dot_spacing + 10
            color = WHITE if i == selected_idx else GRAY
            radius = 6 if i == selected_idx else 4
            pygame.draw.circle(screen, color, (dot_x, dots_y), radius)
        
        # Contador
        counter_text = font.render(f"{selected_idx + 1} / {len(map_list)}", True, GRAY)
        counter_rect = counter_text.get_rect(center=(SCREEN_WIDTH // 2, dots_y + 25))
        screen.blit(counter_text, counter_rect)
        
        # ========================================
        # OPÇÃO DE MAPA GRANDE COM FOG OF WAR
        # ========================================
        large_map_y = dots_y + 60
        
        # Box de opção
        large_map_box = pygame.Rect(SCREEN_WIDTH // 2 - 250, large_map_y, 500, 70)
        
        if self.large_map_enabled:
            pygame.draw.rect(screen, (40, 60, 40), large_map_box, border_radius=10)
            pygame.draw.rect(screen, (100, 255, 100), large_map_box, 3, border_radius=10)
            status_text = "ATIVADO ✓"
            status_color = (100, 255, 100)
        else:
            pygame.draw.rect(screen, (40, 40, 50), large_map_box, border_radius=10)
            pygame.draw.rect(screen, (100, 100, 100), large_map_box, 2, border_radius=10)
            status_text = "DESATIVADO"
            status_color = (150, 150, 150)
        
        # Título da opção
        large_title = font_large.render("🗺️ MAPA GRANDE COM FOG OF WAR", True, WHITE)
        large_rect = large_title.get_rect(center=(SCREEN_WIDTH // 2, large_map_y + 20))
        screen.blit(large_title, large_rect)
        
        # Status
        status_surf = font.render(f"[L] para alternar - {status_text}", True, status_color)
        status_rect = status_surf.get_rect(center=(SCREEN_WIDTH // 2, large_map_y + 48))
        screen.blit(status_surf, status_rect)
        
        # Descrição
        if self.large_map_enabled:
            desc_lines = [
                "Mapa 3000x3000 • Obstáculos • Fog of War por classe",
                f"Campo de visão varia por classe (Ranger: 400, Cleric: 200, etc.)"
            ]
        else:
            desc_lines = [
                "Mapa padrão • Sem obstáculos • Visão total"
            ]
        
        for i, line in enumerate(desc_lines):
            desc_surf = font.render(line, True, (180, 180, 180))
            desc_rect = desc_surf.get_rect(center=(SCREEN_WIDTH // 2, large_map_y + 90 + i * 20))
            screen.blit(desc_surf, desc_rect)
    
    def draw_ai_model_selection(self):
        """Desenha a seleção de modelo de IA treinada"""
        # Título
        title = font_large.render("Selecione o tipo de IA para o inimigo", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 160))
        screen.blit(title, title_rect)
        
        # Opções
        options = ["🎮 IA Simples (Padrão)"]
        
        # Adicionar modelos treinados
        for model in self.game_ai_models:
            agent_class = model.get('agent_class', 'warrior')
            agent_weapon = model.get('agent_weapon', 'sword')
            episode = model.get('episode', 0)
            win_rate = model.get('win_rate', 0)
            options.append(f"🧠 {agent_class.capitalize()} + {agent_weapon.capitalize()} (Ep:{episode}, WR:{win_rate:.0f}%)")
        
        # Calcular posições
        start_y = 220
        item_height = 50
        
        for i, option_text in enumerate(options):
            y = start_y + i * item_height
            is_selected = (i == self.selected_ai_model_idx)
            
            # Box
            box_rect = pygame.Rect(SCREEN_WIDTH // 2 - 300, y, 600, 40)
            
            if is_selected:
                pygame.draw.rect(screen, (60, 80, 60), box_rect)
                pygame.draw.rect(screen, (100, 255, 100), box_rect, 3)
            else:
                pygame.draw.rect(screen, (40, 40, 50), box_rect)
                pygame.draw.rect(screen, GRAY, box_rect, 1)
            
            # Texto
            text_color = (100, 255, 100) if is_selected else WHITE
            text_surf = font_large.render(option_text, True, text_color)
            text_rect = text_surf.get_rect(center=box_rect.center)
            screen.blit(text_surf, text_rect)
        
        # Se não houver modelos
        if len(self.game_ai_models) == 0:
            no_models_text = font.render("Nenhum modelo treinado encontrado. Treine uma IA no menu principal!", True, (255, 200, 100))
            no_models_rect = no_models_text.get_rect(center=(SCREEN_WIDTH // 2, start_y + len(options) * item_height + 30))
            screen.blit(no_models_text, no_models_rect)
        
        # Informação extra sobre o modelo selecionado
        if self.selected_ai_model_idx > 0 and self.selected_ai_model_idx <= len(self.game_ai_models):
            model = self.game_ai_models[self.selected_ai_model_idx - 1]
            
            info_y = start_y + len(options) * item_height + 40
            
            info_lines = [
                f"Treinado contra: {model.get('opponent_class', '?').capitalize()} + {model.get('opponent_weapon', '?').capitalize()}",
                f"Episódios: {model.get('episode', 0)} | Vitórias: {model.get('wins', 0)} | Derrotas: {model.get('losses', 0)}",
                f"Recompensa média: {model.get('avg_reward', 0):.2f} | Melhor: {model.get('best_reward', 0):.2f}"
            ]
            
            for j, line in enumerate(info_lines):
                info_surf = font.render(line, True, (180, 180, 180))
                info_rect = info_surf.get_rect(center=(SCREEN_WIDTH // 2, info_y + j * 25))
                screen.blit(info_surf, info_rect)
    
    def draw_game(self):
        """Desenha a arena e entidades"""
        # Atualizar tempo do mapa para animações
        self.map_time += 1/60
        
        # Determinar se há IA no jogo (para transparência do Assassino)
        has_ai_opponent = self.mode in ["pve", "ai_vs_ai"]
        
        # Sistema de mapa grande com fog of war
        if self.large_map_enabled and self.camera and self.fog_of_war:
            self._draw_large_map_game(has_ai_opponent)
        else:
            # Desenho normal para mapa pequeno
            self._draw_normal_game(has_ai_opponent)
        
        # UI (sempre por cima)
        self.draw_ui()
        
        # Indicador de slow motion
        if self.slow_motion:
            pulse = int(abs(math.sin(pygame.time.get_ticks() / 100)) * 100) + 100
            border_color = (pulse, pulse // 2, 0)
            pygame.draw.rect(screen, border_color, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 8)
            
            slow_text = font_large.render("⚡ SLOW MOTION ⚡", True, (255, 200, 100))
            slow_rect = slow_text.get_rect(center=(SCREEN_WIDTH // 2, 30))
            screen.blit(slow_text, slow_rect)
        
        # Mensagens de estado
        if self.paused:
            self.draw_overlay("PAUSADO", "Pressione P para continuar")
        elif self.game_over:
            if self.winner:
                winner_idx = self.entities.index(self.winner) + 1 if self.winner in self.entities else 0
                self.draw_overlay(f"Jogador {winner_idx} Venceu!", "Pressione R para reiniciar")
            else:
                self.draw_overlay("Empate!", "Pressione R para reiniciar")
    
    def _draw_normal_game(self, has_ai_opponent: bool):
        """Desenha o jogo normal (mapa pequeno)"""
        # Desenhar mapa com tema selecionado
        self.map_renderer.draw(screen, self.current_map, self.map_time)
        
        # Entidades
        for entity in self.entities:
            if hasattr(entity, 'invisible'):
                entity.draw(screen, against_ai=has_ai_opponent)
            else:
                entity.draw(screen)
    
    def _draw_large_map_game(self, has_ai_opponent: bool):
        """Desenha o jogo com mapa grande e fog of war"""
        # Obter offset da câmera
        cam_offset_x, cam_offset_y = self.camera.get_offset()
        
        # Fundo do mapa grande
        screen.fill((20, 30, 20))  # Verde escuro como grama
        
        # Desenhar grid de referência (para mostrar o tamanho do mapa)
        grid_size = 200
        for x in range(0, self.large_map_width, grid_size):
            screen_x = x - cam_offset_x
            if -10 < screen_x < SCREEN_WIDTH + 10:
                pygame.draw.line(screen, (30, 40, 30), (screen_x, 0), (screen_x, SCREEN_HEIGHT), 1)
        for y in range(0, self.large_map_height, grid_size):
            screen_y = y - cam_offset_y
            if -10 < screen_y < SCREEN_HEIGHT + 10:
                pygame.draw.line(screen, (30, 40, 30), (0, screen_y), (SCREEN_WIDTH, screen_y), 1)
        
        # Desenhar bordas do mapa
        border_rect = pygame.Rect(
            -cam_offset_x, -cam_offset_y,
            self.large_map_width, self.large_map_height
        )
        pygame.draw.rect(screen, (100, 100, 100), border_rect, 3)
        
        # Desenhar obstáculos
        if self.obstacle_manager:
            self.obstacle_manager.draw(screen, (cam_offset_x, cam_offset_y))
        
        # Obter visão do jogador que controla a câmera
        viewer_class = "warrior"  # Default
        if self.camera_target:
            viewer_class = self.camera_target.__class__.__name__.lower()
        
        # Desenhar entidades (com fog of war)
        for entity in self.entities:
            # Converter posição do mundo para tela
            screen_x = entity.x - cam_offset_x
            screen_y = entity.y - cam_offset_y
            
            # Só desenhar se estiver na tela
            if -50 < screen_x < SCREEN_WIDTH + 50 and -50 < screen_y < SCREEN_HEIGHT + 50:
                # Verificar se está visível (para o fog of war)
                is_visible = True
                if entity != self.camera_target and self.camera_target:
                    # Verificar linha de visão usando o FogOfWar
                    is_visible = self.fog_of_war.is_entity_visible_by(
                        self.camera_target, entity
                    )
                
                if is_visible:
                    # Desenhar entidade com offset da câmera
                    self._draw_entity_with_offset(entity, screen_x, screen_y, has_ai_opponent)
                else:
                    # Desenhar indicador de última posição conhecida (opcional)
                    pass
        
        # Aplicar fog of war overlay
        viewer_team = self.camera_target.team if self.camera_target and hasattr(self.camera_target, 'team') else "none"
        if viewer_team != "none":
            self.fog_of_war.draw_fog(screen, viewer_team, self.entities, (cam_offset_x, cam_offset_y))
        
        # Minimapa no canto
        self._draw_minimap()
    
    def _draw_entity_with_offset(self, entity, screen_x: float, screen_y: float, against_ai: bool):
        """Desenha uma entidade com offset da câmera"""
        # Salvar posição original
        original_x = entity.x
        original_y = entity.y
        
        # Temporariamente mover para posição na tela
        entity.x = screen_x
        entity.y = screen_y
        
        # Desenhar
        if hasattr(entity, 'invisible'):
            entity.draw(screen, against_ai=against_ai)
        else:
            entity.draw(screen)
        
        # Restaurar posição original
        entity.x = original_x
        entity.y = original_y
    
    def _draw_minimap(self):
        """Desenha um minimapa no canto da tela"""
        minimap_width = 180
        minimap_height = 180
        minimap_x = SCREEN_WIDTH - minimap_width - 10
        minimap_y = SCREEN_HEIGHT - minimap_height - 10
        
        # Fundo do minimapa
        minimap_surface = pygame.Surface((minimap_width, minimap_height), pygame.SRCALPHA)
        minimap_surface.fill((0, 0, 0, 180))
        
        # Escala
        scale_x = minimap_width / self.large_map_width
        scale_y = minimap_height / self.large_map_height
        
        # Desenhar obstáculos no minimapa
        if self.obstacle_manager:
            for obs in self.obstacle_manager.obstacles:
                obs_x = int(obs.rect.x * scale_x)
                obs_y = int(obs.rect.y * scale_y)
                obs_w = max(2, int(obs.rect.width * scale_x))
                obs_h = max(2, int(obs.rect.height * scale_y))
                pygame.draw.rect(minimap_surface, (80, 80, 80), (obs_x, obs_y, obs_w, obs_h))
        
        # Desenhar entidades no minimapa
        for entity in self.entities:
            if entity.is_alive():
                ent_x = int(entity.x * scale_x)
                ent_y = int(entity.y * scale_y)
                color = entity.color if entity == self.camera_target else (150, 150, 150)
                pygame.draw.circle(minimap_surface, color, (ent_x, ent_y), 4)
        
        # Desenhar viewport atual (usando offset da câmera)
        if self.camera:
            cam_offset_x, cam_offset_y = self.camera.get_offset()
            vp_x = int(cam_offset_x * scale_x)
            vp_y = int(cam_offset_y * scale_y)
            vp_w = int(SCREEN_WIDTH * scale_x)
            vp_h = int(SCREEN_HEIGHT * scale_y)
            pygame.draw.rect(minimap_surface, (255, 255, 255), (vp_x, vp_y, vp_w, vp_h), 1)
        
        # Borda do minimapa
        pygame.draw.rect(minimap_surface, (100, 100, 100), (0, 0, minimap_width, minimap_height), 2)
        
        # Blitar minimapa na tela
        screen.blit(minimap_surface, (minimap_x, minimap_y))
    
    def draw_ui(self):
        """Desenha a interface do usuário"""
        y_offset = 10
        
        for i, entity in enumerate(self.entities):
            # Info do jogador
            stats = entity.stats_manager.get_stats()
            class_name = entity.display_name
            
            info_text = f"P{i+1} ({class_name}): {int(entity.health)}/{int(stats.max_health)} HP"
            text_surf = font.render(info_text, True, entity.color)
            screen.blit(text_surf, (10, y_offset))
            
            # Barra de vida
            bar_width = 150
            bar_height = 10
            health_ratio = entity.health / stats.max_health
            
            pygame.draw.rect(screen, (60, 60, 60), 
                           (10, y_offset + 22, bar_width, bar_height))
            pygame.draw.rect(screen, entity.color, 
                           (10, y_offset + 22, bar_width * health_ratio, bar_height))
            pygame.draw.rect(screen, WHITE, 
                           (10, y_offset + 22, bar_width, bar_height), 1)
            
            # Cooldown de habilidade
            ability_info = entity.get_ability_info()
            if ability_info['cooldown'] > 0:
                if entity.ability_cooldown > 0:
                    cd_text = f"[{ability_info['name']}]: {entity.ability_cooldown:.1f}s"
                    cd_color = (150, 150, 150)
                else:
                    cd_text = f"[{ability_info['name']}]: Pronto!"
                    cd_color = (100, 255, 100)
                
                cd_surf = font.render(cd_text, True, cd_color)
                screen.blit(cd_surf, (10, y_offset + 36))
            
            y_offset += 60
        
        # Instruções
        instructions = [
            "P1: WASD + ESPAÇO (ataque) + SHIFT (habilidade)",
            "P2: Setas + ENTER (ataque) + R.SHIFT (habilidade)",
            "R - Reiniciar | P - Pausar | ESC - Menu"
        ]
        
        for i, text in enumerate(instructions):
            surf = font.render(text, True, WHITE)
            screen.blit(surf, (SCREEN_WIDTH - surf.get_width() - 10, 10 + i * 20))
    
    def draw_overlay(self, title: str, subtitle: str):
        """Desenha uma sobreposição com mensagem"""
        # Fundo semi-transparente
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        
        # Título
        title_surf = font_title.render(title, True, WHITE)
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        screen.blit(title_surf, title_rect)
        
        # Subtítulo
        sub_surf = font_large.render(subtitle, True, GRAY)
        sub_rect = sub_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
        screen.blit(sub_surf, sub_rect)
    
    # ==========================================================================
    # DESENHO DO MENU DE BATALHA EM GRUPO
    # ==========================================================================
    
    def draw_group_menu(self):
        """Desenha o menu de batalha em grupo"""
        # Título
        title = font_title.render("⚔️ BATALHA EM GRUPO ⚔️", True, (255, 150, 255))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 60))
        screen.blit(title, title_rect)
        
        if self.group_menu_state == "main":
            self._draw_group_main_menu()
        elif self.group_menu_state == "select_size":
            self._draw_group_size_select()
        elif self.group_menu_state in ["select_blue", "select_red"]:
            self._draw_group_composition_select()
        elif self.group_menu_state == "stats":
            self._draw_group_stats()
        elif self.group_menu_state == "train_setup":
            self._draw_group_train_setup()
        
        # Instruções
        instructions = font.render("ESC - Voltar", True, GRAY)
        screen.blit(instructions, (50, SCREEN_HEIGHT - 40))
    
    def _draw_group_main_menu(self):
        """Desenha o menu principal de grupo"""
        subtitle = font_large.render("Combate cooperativo com IA inteligente", True, GRAY)
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(subtitle, sub_rect)
        
        # Box informativo
        info_rect = pygame.Rect(100, 140, SCREEN_WIDTH - 200, 100)
        pygame.draw.rect(screen, (40, 40, 50), info_rect, border_radius=10)
        pygame.draw.rect(screen, (100, 80, 150), info_rect, 2, border_radius=10)
        
        info_lines = [
            "Crie equipes com diferentes papéis: Tank, Healer, DPS, Controller",
            "Cada papel tem comportamento e estratégia únicos!",
            "Treine redes neurais que entendem seu papel no grupo"
        ]
        for i, line in enumerate(info_lines):
            surf = font.render(line, True, (200, 200, 220))
            screen.blit(surf, (120, 155 + i * 28))
        
        # Opções
        options = [
            ("1 - Batalha Rápida", 290, (100, 200, 255)),
            ("2 - Treinar Grupo 🧠", 340, (100, 255, 150)),
            ("3 - Estatísticas", 390, (255, 200, 100)),
        ]
        
        for text, y, color in options:
            surf = font_large.render(text, True, color)
            rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
            screen.blit(surf, rect)
        
        # Stats rápidos
        stats_y = 470
        pygame.draw.line(screen, GRAY, (100, stats_y - 20), (SCREEN_WIDTH - 100, stats_y - 20), 1)
        
        stats_title = font_large.render("📊 Estatísticas da Sessão", True, (200, 150, 255))
        screen.blit(stats_title, (100, stats_y))
        
        stats_text = f"Batalhas: {self.group_battle_stats['total_battles']} | " \
                    f"Azul: {self.group_battle_stats['blue_wins']} | " \
                    f"Vermelho: {self.group_battle_stats['red_wins']} | " \
                    f"Empates: {self.group_battle_stats['draws']}"
        stats_surf = font.render(stats_text, True, (180, 180, 180))
        screen.blit(stats_surf, (100, stats_y + 35))
    
    def _draw_group_size_select(self):
        """Desenha seleção de tamanho de equipe"""
        subtitle = font_large.render("Escolha o tamanho das equipes", True, GRAY)
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 120))
        screen.blit(subtitle, sub_rect)
        
        sizes = [
            ("1 - Batalha 2v2", 220, "Duelo em duplas - rápido e tático", (100, 200, 255)),
            ("2 - Batalha 3v3", 300, "Trios - equilíbrio ideal", (150, 200, 150)),
            ("3 - Batalha 5v5", 380, "Equipes completas - estratégia total", (255, 180, 100)),
        ]
        
        for text, y, desc, color in sizes:
            # Título
            surf = font_large.render(text, True, color)
            rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
            screen.blit(surf, rect)
            
            # Descrição
            desc_surf = font.render(desc, True, GRAY)
            desc_rect = desc_surf.get_rect(center=(SCREEN_WIDTH // 2, y + 30))
            screen.blit(desc_surf, desc_rect)
    
    def _draw_group_composition_select(self):
        """Desenha seleção de composição de equipe"""
        team_name = "TIME AZUL" if self.group_menu_state == "select_blue" else "TIME VERMELHO"
        team_color = (80, 140, 255) if self.group_menu_state == "select_blue" else (255, 80, 80)
        
        subtitle = font_large.render(f"Selecione composição para {team_name}", True, team_color)
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 110))
        screen.blit(subtitle, sub_rect)
        
        size_text = font.render(f"Modo: {self.team_size}v{self.team_size}", True, GRAY)
        screen.blit(size_text, (SCREEN_WIDTH // 2 - size_text.get_width() // 2, 140))
        
        # Lista de composições
        compositions = self._get_compositions_for_size(self.team_size)
        comp_keys = list(compositions.keys())
        
        start_y = 190
        for i, key in enumerate(comp_keys):
            comp = compositions[key]
            y = start_y + i * 90
            
            is_selected = (i == self.group_selected_composition)
            
            # Box
            box_rect = pygame.Rect(150, y - 10, SCREEN_WIDTH - 300, 80)
            if is_selected:
                pygame.draw.rect(screen, (50, 60, 80), box_rect, border_radius=10)
                pygame.draw.rect(screen, team_color, box_rect, 3, border_radius=10)
            else:
                pygame.draw.rect(screen, (35, 35, 45), box_rect, border_radius=10)
                pygame.draw.rect(screen, (80, 80, 80), box_rect, 1, border_radius=10)
            
            # Nome da composição
            name = key.replace("_", " ").title()
            name_color = WHITE if is_selected else (150, 150, 150)
            name_surf = font_large.render(name, True, name_color)
            screen.blit(name_surf, (170, y))
            
            # Membros
            members = []
            for member in comp:
                role_colors = {
                    "dps_melee": (255, 150, 100),
                    "dps_ranged": (255, 200, 100),
                    "tank": (100, 150, 200),
                    "healer": (100, 255, 150),
                    "controller": (200, 100, 255),
                    "support": (255, 220, 100),
                }
                role = member.get("role", "dps_melee")
                role_color = role_colors.get(role, (180, 180, 180))
                members.append((member["class"].title(), role_color))
            
            # Desenhar membros
            x_offset = 170
            for member_name, color in members:
                member_surf = font.render(f"• {member_name}", True, color)
                screen.blit(member_surf, (x_offset, y + 35))
                x_offset += member_surf.get_width() + 20
        
        # Instruções
        nav_text = font.render("↑↓ Navegar | ENTER Confirmar | ESC Voltar", True, GRAY)
        screen.blit(nav_text, (SCREEN_WIDTH // 2 - nav_text.get_width() // 2, SCREEN_HEIGHT - 70))
    
    def _draw_group_stats(self):
        """Desenha estatísticas de grupo"""
        subtitle = font_large.render("📊 Estatísticas Detalhadas", True, (255, 200, 100))
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 120))
        screen.blit(subtitle, sub_rect)
        
        stats = self.group_battle_stats
        total = max(1, stats["total_battles"])
        
        stats_lines = [
            (f"Total de Batalhas: {stats['total_battles']}", WHITE),
            (f"Vitórias Azul: {stats['blue_wins']} ({100*stats['blue_wins']/total:.1f}%)", (80, 140, 255)),
            (f"Vitórias Vermelho: {stats['red_wins']} ({100*stats['red_wins']/total:.1f}%)", (255, 80, 80)),
            (f"Empates: {stats['draws']} ({100*stats['draws']/total:.1f}%)", (150, 150, 150)),
        ]
        
        y = 200
        for text, color in stats_lines:
            surf = font_large.render(text, True, color)
            screen.blit(surf, (SCREEN_WIDTH // 2 - surf.get_width() // 2, y))
            y += 50
    
    def _draw_group_train_setup(self):
        """Desenha configuração de treinamento em grupo"""
        subtitle = font_large.render("🧠 Treinamento de IA em Grupo", True, (100, 255, 150))
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 120))
        screen.blit(subtitle, sub_rect)
        
        info_text = "Em desenvolvimento - A IA aprenderá seu papel no grupo!"
        info_surf = font.render(info_text, True, GRAY)
        screen.blit(info_surf, (SCREEN_WIDTH // 2 - info_surf.get_width() // 2, 200))
        
        # Explicação dos papéis
        roles_info = [
            ("🗡️ DPS", "Foca em causar dano e eliminar inimigos", (255, 150, 100)),
            ("🛡️ Tank", "Protege aliados e absorve dano", (100, 150, 200)),
            ("💚 Healer", "Mantém a equipe viva", (100, 255, 150)),
            ("⚡ Controller", "Controla inimigos com CC", (200, 100, 255)),
            ("✨ Support", "Buffa aliados e debuffa inimigos", (255, 220, 100)),
        ]
        
        y = 280
        for name, desc, color in roles_info:
            name_surf = font_large.render(name, True, color)
            screen.blit(name_surf, (200, y))
            
            desc_surf = font.render(desc, True, (180, 180, 180))
            screen.blit(desc_surf, (350, y + 5))
            y += 45
    
    def draw_group_battle(self):
        """Desenha a batalha em grupo"""
        # Desenhar mapa
        self.map_renderer.draw(screen, self.current_map, self.map_time)
        
        # Linha central
        pygame.draw.line(
            screen, (60, 60, 80),
            (SCREEN_WIDTH // 2, self.arena_config.border),
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT - self.arena_config.border),
            2
        )
        
        # Desenhar entidades
        for entity in self.entities:
            if entity.is_alive():
                entity.draw(screen)
        
        # UI
        self._draw_group_battle_ui()
        
        # Overlay se jogo acabou
        if self.game_over:
            self._draw_group_battle_result()
    
    def _draw_group_battle_ui(self):
        """Desenha UI da batalha em grupo"""
        # Placar superior
        blue_alive = sum(1 for e in self.blue_team if e.is_alive())
        red_alive = sum(1 for e in self.red_team if e.is_alive())
        
        # Background do placar
        score_bg = pygame.Rect(SCREEN_WIDTH // 2 - 150, 5, 300, 40)
        pygame.draw.rect(screen, (30, 30, 40), score_bg, border_radius=8)
        
        # Time Azul
        blue_text = font_large.render(f"🔵 {blue_alive}", True, (80, 140, 255))
        screen.blit(blue_text, (SCREEN_WIDTH // 2 - 120, 12))
        
        # VS
        vs_text = font_large.render("VS", True, WHITE)
        screen.blit(vs_text, (SCREEN_WIDTH // 2 - vs_text.get_width() // 2, 12))
        
        # Time Vermelho
        red_text = font_large.render(f"{red_alive} 🔴", True, (255, 80, 80))
        screen.blit(red_text, (SCREEN_WIDTH // 2 + 60, 12))
        
        # Barras de vida individuais
        self._draw_team_health_bars()
        
        # Controles
        controls = font.render("P: Pausar | R: Reiniciar | 1/2/4: Velocidade | ESC: Sair", True, GRAY)
        screen.blit(controls, (10, SCREEN_HEIGHT - 25))
        
        # Velocidade
        speed_text = font.render(f"Velocidade: {self.training_speed}x", True, (150, 150, 150))
        screen.blit(speed_text, (SCREEN_WIDTH - speed_text.get_width() - 10, SCREEN_HEIGHT - 25))
        
        # Pausado
        if self.paused:
            pause_surf = font_title.render("PAUSADO", True, (255, 255, 100))
            pause_rect = pause_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(pause_surf, pause_rect)
    
    def _draw_team_health_bars(self):
        """Desenha barras de vida das equipes"""
        bar_width = 80
        bar_height = 8
        
        # Time Azul (esquerda)
        x_blue = 10
        y = 60
        for entity in self.blue_team:
            if entity.is_alive():
                health_pct = entity.health / entity.stats_manager.get_stats().max_health
                
                # Fundo
                pygame.draw.rect(screen, (50, 50, 50), (x_blue, y, bar_width, bar_height))
                # Vida
                pygame.draw.rect(screen, (80, 200, 80), (x_blue, y, bar_width * health_pct, bar_height))
                # Nome
                name = entity.__class__.__name__[:8]
                name_surf = font.render(name, True, (80, 140, 255))
                screen.blit(name_surf, (x_blue, y - 18))
                
                y += 35
        
        # Time Vermelho (direita)
        x_red = SCREEN_WIDTH - bar_width - 10
        y = 60
        for entity in self.red_team:
            if entity.is_alive():
                health_pct = entity.health / entity.stats_manager.get_stats().max_health
                
                pygame.draw.rect(screen, (50, 50, 50), (x_red, y, bar_width, bar_height))
                pygame.draw.rect(screen, (80, 200, 80), (x_red, y, bar_width * health_pct, bar_height))
                
                name = entity.__class__.__name__[:8]
                name_surf = font.render(name, True, (255, 80, 80))
                screen.blit(name_surf, (x_red, y - 18))
                
                y += 35
    
    def _draw_group_battle_result(self):
        """Desenha resultado da batalha"""
        # Overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        
        # Resultado
        if self.winner == "blue":
            result_text = "🔵 VITÓRIA AZUL! 🔵"
            result_color = (80, 140, 255)
        elif self.winner == "red":
            result_text = "🔴 VITÓRIA VERMELHA! 🔴"
            result_color = (255, 80, 80)
        else:
            result_text = "⚔️ EMPATE! ⚔️"
            result_color = (200, 200, 100)
        
        result_surf = font_title.render(result_text, True, result_color)
        result_rect = result_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        screen.blit(result_surf, result_rect)
        
        hint_surf = font_large.render("R - Revanche | ESC - Menu", True, GRAY)
        hint_rect = hint_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
        screen.blit(hint_surf, hint_rect)

    def draw_training_menu(self):
        """Desenha o menu de treinamento"""
        # Se o menu de carregar está ativo, desenhar ele
        if self.load_menu_active:
            self.draw_load_menu()
            return
        
        # Título
        title = font_title.render("🧠 TREINAMENTO DE IA", True, (100, 255, 150))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
        screen.blit(title, title_rect)
        
        subtitle = font_large.render("Machine Learning com Reinforcement Learning", True, GRAY)
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 120))
        screen.blit(subtitle, sub_rect)
        
        # Box informativo
        info_rect = pygame.Rect(100, 160, SCREEN_WIDTH - 200, 120)
        pygame.draw.rect(screen, (40, 40, 40), info_rect, border_radius=10)
        pygame.draw.rect(screen, (80, 80, 80), info_rect, 2, border_radius=10)
        
        info_lines = [
            "O treinamento usa Reinforcement Learning (PPO) para ensinar",
            "uma rede neural a jogar. A IA aprende através de tentativa",
            "e erro, recebendo recompensas por acertar ataques e vencer.",
            "Cada combinação de classe/arma cria estratégias diferentes!"
        ]
        
        for i, line in enumerate(info_lines):
            surf = font.render(line, True, (200, 200, 200))
            screen.blit(surf, (120, 175 + i * 25))
        
        # Opções
        options = [
            ("1 - Novo Treinamento", 320, (100, 255, 150)),
            ("2 - Carregar Modelo", 360, (100, 200, 255)),
            ("3 - Testar Modelo", 400, (255, 200, 100)),
            ("4 - Ver Estatísticas dos Modelos", 440, (200, 150, 255)),
        ]
        
        for text, y, color in options:
            surf = font_large.render(text, True, color)
            rect = surf.get_rect(center=(SCREEN_WIDTH // 2, y))
            screen.blit(surf, rect)
        
        # Modelos salvos (preview)
        pygame.draw.line(screen, GRAY, (100, 490), (SCREEN_WIDTH - 100, 490), 2)
        
        models_title = font_large.render("Modelos Salvos:", True, (255, 200, 100))
        screen.blit(models_title, (100, 510))
        
        if os.path.exists("models"):
            models = [f for f in os.listdir("models") if f.endswith('.json')]
            if models:
                for i, model in enumerate(models[:5]):  # Máximo 5 modelos
                    color = (180, 180, 180)
                    surf = font.render(f"• {model}", True, color)
                    screen.blit(surf, (120, 545 + i * 25))
                if len(models) > 5:
                    more_surf = font.render(f"  ... e mais {len(models) - 5} modelos", True, (100, 100, 100))
                    screen.blit(more_surf, (120, 545 + 5 * 25))
            else:
                surf = font.render("Nenhum modelo salvo ainda", True, (100, 100, 100))
                screen.blit(surf, (120, 545))
        else:
            surf = font.render("Pasta 'models' não existe", True, (100, 100, 100))
            screen.blit(surf, (120, 545))
        
        # Instruções
        instructions = font.render("ESC - Voltar ao Menu", True, GRAY)
        screen.blit(instructions, (50, SCREEN_HEIGHT - 40))
    
    def draw_load_menu(self):
        """Desenha o menu de carregar modelos"""
        # Título
        title = font_title.render("📂 Carregar Modelo", True, (100, 200, 255))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 60))
        screen.blit(title, title_rect)
        
        if not self.available_models:
            # Nenhum modelo disponível
            no_model = font_large.render("Nenhum modelo salvo encontrado!", True, (255, 100, 100))
            no_rect = no_model.get_rect(center=(SCREEN_WIDTH // 2, 300))
            screen.blit(no_model, no_rect)
            
            hint = font.render("Treine um modelo primeiro e salve com S", True, GRAY)
            hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, 350))
            screen.blit(hint, hint_rect)
        else:
            # Lista de modelos
            list_rect = pygame.Rect(100, 100, SCREEN_WIDTH - 200, 550)
            pygame.draw.rect(screen, (35, 35, 35), list_rect, border_radius=10)
            pygame.draw.rect(screen, (80, 80, 80), list_rect, 2, border_radius=10)
            
            for i, model in enumerate(self.available_models[:12]):  # Máximo 12
                y = 120 + i * 45
                
                is_selected = (i == self.selected_model_idx)
                
                # Background do item
                item_rect = pygame.Rect(110, y - 5, SCREEN_WIDTH - 220, 40)
                if is_selected:
                    pygame.draw.rect(screen, (60, 80, 100), item_rect, border_radius=5)
                    pygame.draw.rect(screen, (100, 200, 255), item_rect, 2, border_radius=5)
                
                # Nome do arquivo
                name_color = WHITE if is_selected else (180, 180, 180)
                name_surf = font_large.render(model.get('filename', 'Unknown'), True, name_color)
                screen.blit(name_surf, (120, y))
                
                # Info do modelo
                info = f"{model.get('agent_class', '?').title()} + {model.get('agent_weapon', '?').title()} | "
                info += f"Win: {model.get('win_rate', 0):.1f}% | Ep: {model.get('episode', 0)}"
                info_color = (150, 200, 255) if is_selected else (120, 120, 120)
                info_surf = font.render(info, True, info_color)
                screen.blit(info_surf, (120, y + 22))
            
            # Preview do modelo selecionado
            if self.available_models:
                model = self.available_models[self.selected_model_idx]
                
                preview_rect = pygame.Rect(SCREEN_WIDTH - 350, 660, 330, 100)
                pygame.draw.rect(screen, (45, 45, 45), preview_rect, border_radius=10)
                
                preview_title = font_large.render("📊 Detalhes", True, (255, 200, 100))
                screen.blit(preview_title, (SCREEN_WIDTH - 340, 665))
                
                details = [
                    f"Agente: {model.get('agent_class', '?').title()} + {model.get('agent_weapon', '?').title()}",
                    f"Oponente: {model.get('opponent_class', '?').title()} + {model.get('opponent_weapon', '?').title()}",
                    f"Vitórias: {model.get('wins', 0)} | Derrotas: {model.get('losses', 0)}"
                ]
                
                for j, detail in enumerate(details):
                    detail_surf = font.render(detail, True, (200, 200, 200))
                    screen.blit(detail_surf, (SCREEN_WIDTH - 340, 695 + j * 20))
        
        # Instruções
        instructions = "↑↓ Navegar | ENTER Carregar | DEL/X Excluir | ESC Voltar"
        inst_surf = font.render(instructions, True, GRAY)
        inst_rect = inst_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        screen.blit(inst_surf, inst_rect)
    
    def draw_model_stats_view(self):
        """Desenha a visualização detalhada de estatísticas dos modelos"""
        # Título
        title = font_title.render("📊 ESTATÍSTICAS DOS MODELOS", True, (255, 200, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title, title_rect)
        
        if not self.available_models:
            no_model = font_large.render("Nenhum modelo salvo encontrado!", True, (255, 100, 100))
            no_rect = no_model.get_rect(center=(SCREEN_WIDTH // 2, 300))
            screen.blit(no_model, no_rect)
            
            hint = font.render("Treine um modelo primeiro", True, GRAY)
            hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, 350))
            screen.blit(hint, hint_rect)
        else:
            # Lista lateral de modelos
            list_rect = pygame.Rect(30, 90, 280, SCREEN_HEIGHT - 140)
            pygame.draw.rect(screen, (35, 35, 35), list_rect, border_radius=10)
            pygame.draw.rect(screen, (80, 80, 80), list_rect, 2, border_radius=10)
            
            list_title = font_large.render("Modelos", True, (150, 150, 150))
            screen.blit(list_title, (50, 100))
            
            for i, model in enumerate(self.available_models[:15]):
                y = 135 + i * 32
                is_selected = (i == self.model_stats_selected)
                
                if is_selected:
                    sel_rect = pygame.Rect(40, y - 3, 260, 28)
                    pygame.draw.rect(screen, (60, 80, 100), sel_rect, border_radius=5)
                    pygame.draw.rect(screen, (100, 200, 255), sel_rect, 2, border_radius=5)
                
                # Nome resumido
                filename = model.get('filename', 'Unknown')
                if len(filename) > 30:
                    filename = filename[:27] + "..."
                
                name_color = WHITE if is_selected else (150, 150, 150)
                name_surf = font.render(filename, True, name_color)
                screen.blit(name_surf, (50, y))
            
            # Painel de detalhes do modelo selecionado
            if self.available_models:
                model = self.available_models[self.model_stats_selected]
                
                detail_rect = pygame.Rect(330, 90, SCREEN_WIDTH - 360, SCREEN_HEIGHT - 140)
                pygame.draw.rect(screen, (40, 40, 45), detail_rect, border_radius=15)
                pygame.draw.rect(screen, (80, 100, 120), detail_rect, 3, border_radius=15)
                
                # Nome do modelo
                model_name = model.get('filename', 'Unknown')
                name_surf = font_large.render(f"📁 {model_name}", True, WHITE)
                screen.blit(name_surf, (350, 110))
                
                # Linha divisória
                pygame.draw.line(screen, (80, 80, 80), (350, 145), (SCREEN_WIDTH - 50, 145), 2)
                
                # Configuração do agente
                agent_title = font_large.render("🤖 Agente", True, (100, 200, 255))
                screen.blit(agent_title, (350, 160))
                
                agent_class = model.get('agent_class', '?').title()
                agent_weapon = model.get('agent_weapon', '?').title()
                agent_info = font.render(f"Classe: {agent_class} | Arma: {agent_weapon}", True, (180, 180, 180))
                screen.blit(agent_info, (370, 190))
                
                # Configuração do oponente
                opp_title = font_large.render("👹 Oponente de Treino", True, (255, 150, 100))
                screen.blit(opp_title, (350, 225))
                
                if model.get('multi_opponent', False):
                    opp_info = font.render("Multi-oponente (20 combinações rotativas)", True, (180, 180, 180))
                else:
                    opp_class = model.get('opponent_class', '?').title()
                    opp_weapon = model.get('opponent_weapon', '?').title()
                    opp_info = font.render(f"Classe: {opp_class} | Arma: {opp_weapon}", True, (180, 180, 180))
                screen.blit(opp_info, (370, 255))
                
                # Estatísticas de treinamento
                pygame.draw.line(screen, (80, 80, 80), (350, 295), (SCREEN_WIDTH - 50, 295), 2)
                stats_title = font_large.render("📈 Estatísticas de Treinamento", True, (100, 255, 150))
                screen.blit(stats_title, (350, 310))
                
                stats_data = [
                    ("Episódio:", str(model.get('episode', 0))),
                    ("Vitórias:", str(model.get('wins', 0))),
                    ("Derrotas:", str(model.get('losses', 0))),
                    ("Taxa de Vitória:", f"{model.get('win_rate', 0):.1f}%"),
                    ("Recompensa Média:", f"{model.get('avg_reward', 0):.2f}"),
                    ("Melhor Recompensa:", f"{model.get('best_reward', 0):.2f}"),
                ]
                
                for i, (label, value) in enumerate(stats_data):
                    row = i // 2
                    col = i % 2
                    x = 370 + col * 250
                    y = 350 + row * 40
                    
                    label_surf = font.render(label, True, (150, 150, 150))
                    screen.blit(label_surf, (x, y))
                    
                    # Colorir valores baseado no desempenho
                    if "Taxa" in label:
                        win_rate = model.get('win_rate', 0)
                        if win_rate >= 70:
                            val_color = (100, 255, 100)
                        elif win_rate >= 40:
                            val_color = (255, 255, 100)
                        else:
                            val_color = (255, 100, 100)
                    else:
                        val_color = (200, 200, 200)
                    
                    val_surf = font_large.render(value, True, val_color)
                    screen.blit(val_surf, (x + 130, y - 3))
                
                # Gráfico visual de desempenho (barra de progresso)
                pygame.draw.line(screen, (80, 80, 80), (350, 480), (SCREEN_WIDTH - 50, 480), 2)
                perf_title = font_large.render("🎯 Desempenho", True, (255, 200, 100))
                screen.blit(perf_title, (350, 495))
                
                # Barra de win rate
                bar_x = 370
                bar_y = 535
                bar_w = SCREEN_WIDTH - 420
                bar_h = 30
                
                # Fundo da barra
                pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, bar_h), border_radius=5)
                
                # Preenchimento
                win_rate = model.get('win_rate', 0)
                fill_w = int((win_rate / 100) * bar_w)
                
                # Cor baseada no desempenho
                if win_rate >= 70:
                    fill_color = (50, 200, 80)
                elif win_rate >= 40:
                    fill_color = (200, 200, 50)
                else:
                    fill_color = (200, 80, 50)
                
                if fill_w > 0:
                    pygame.draw.rect(screen, fill_color, (bar_x, bar_y, fill_w, bar_h), border_radius=5)
                
                # Texto da porcentagem
                rate_text = font_large.render(f"{win_rate:.1f}%", True, WHITE)
                rate_rect = rate_text.get_rect(center=(bar_x + bar_w // 2, bar_y + bar_h // 2))
                screen.blit(rate_text, rate_rect)
                
                # Win/Loss em texto
                wins = model.get('wins', 0)
                losses = model.get('losses', 0)
                total = wins + losses
                record_text = font.render(f"Histórico: {wins}W - {losses}L ({total} jogos)", True, (150, 150, 150))
                screen.blit(record_text, (bar_x, bar_y + 40))
                
                # Timestamp se disponível
                if model.get('timestamp'):
                    time_text = font.render(f"Salvo em: {model.get('timestamp', 'N/A')}", True, (100, 100, 100))
                    screen.blit(time_text, (350, SCREEN_HEIGHT - 80))
        
        # Instruções
        instr_bg = pygame.Surface((SCREEN_WIDTH, 40), pygame.SRCALPHA)
        instr_bg.fill((30, 30, 40, 200))
        screen.blit(instr_bg, (0, SCREEN_HEIGHT - 50))
        
        instructions = "↑↓ Navegar | DEL/X Excluir Modelo | ESC Voltar"
        inst_surf = font.render(instructions, True, GRAY)
        inst_rect = inst_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        screen.blit(inst_surf, inst_rect)
        
        # Mensagem de status
        if self.save_message_timer > 0:
            msg_surf = font_large.render(self.save_message, True, (255, 255, 100))
            msg_rect = msg_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 70))
            screen.blit(msg_surf, msg_rect)

    def draw_model_delete_confirm(self):
        """Desenha a confirmação de exclusão de modelo"""
        # Overlay escuro
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        
        # Caixa de diálogo
        dialog_w = 500
        dialog_h = 200
        dialog_x = (SCREEN_WIDTH - dialog_w) // 2
        dialog_y = (SCREEN_HEIGHT - dialog_h) // 2
        
        pygame.draw.rect(screen, (50, 50, 60), (dialog_x, dialog_y, dialog_w, dialog_h), border_radius=15)
        pygame.draw.rect(screen, (255, 100, 100), (dialog_x, dialog_y, dialog_w, dialog_h), 3, border_radius=15)
        
        # Título
        title = font_large.render("⚠️ Confirmar Exclusão", True, (255, 100, 100))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, dialog_y + 40))
        screen.blit(title, title_rect)
        
        # Mensagem
        msg = font.render("Tem certeza que deseja excluir este modelo?", True, WHITE)
        msg_rect = msg.get_rect(center=(SCREEN_WIDTH // 2, dialog_y + 90))
        screen.blit(msg, msg_rect)
        
        msg2 = font.render("Esta ação não pode ser desfeita!", True, (255, 200, 100))
        msg2_rect = msg2.get_rect(center=(SCREEN_WIDTH // 2, dialog_y + 115))
        screen.blit(msg2, msg2_rect)
        
        # Botões
        yes_surf = font_large.render("[ Y ] Sim, Excluir", True, (255, 100, 100))
        no_surf = font_large.render("[ N ] Não, Cancelar", True, (100, 255, 100))
        
        screen.blit(yes_surf, (dialog_x + 50, dialog_y + 150))
        screen.blit(no_surf, (dialog_x + 280, dialog_y + 150))
    
    def draw_training_selection(self):
        """Desenha a tela de seleção para treinamento"""
        # Título
        title = font_title.render("🎯 Configurar Treinamento", True, (100, 255, 150))
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title, title_rect)
        
        # Etapa atual
        if self.training_multi_opponent:
            steps = {
                "agent_class": "1/4 - Classe do Agente (IA)",
                "agent_weapon": "2/4 - Arma do Agente",
                "opponent_type": "3/4 - Tipo de Oponente",
                "config": "4/4 - Configurações"
            }
        else:
            steps = {
                "agent_class": "1/6 - Classe do Agente (IA)",
                "agent_weapon": "2/6 - Arma do Agente",
                "opponent_type": "3/6 - Tipo de Oponente",
                "opponent_class": "4/6 - Classe do Oponente",
                "opponent_weapon": "5/6 - Arma do Oponente",
                "config": "6/6 - Configurações"
            }
        
        step_text = steps.get(self.training_selection_step, "")
        step_surf = font_large.render(step_text, True, WHITE)
        step_rect = step_surf.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(step_surf, step_rect)
        
        # Área de seleção principal
        main_area = pygame.Rect(50, 140, SCREEN_WIDTH - 100, 350)
        pygame.draw.rect(screen, (35, 35, 35), main_area, border_radius=15)
        pygame.draw.rect(screen, (80, 80, 80), main_area, 2, border_radius=15)
        
        if self.training_selection_step in ["agent_class", "opponent_class"]:
            self._draw_class_selector(
                self.training_class if "agent" in self.training_selection_step else self.training_opponent_class,
                "agent" if "agent" in self.training_selection_step else "opponent"
            )
        elif self.training_selection_step in ["agent_weapon", "opponent_weapon"]:
            self._draw_weapon_selector(
                self.training_weapon if "agent" in self.training_selection_step else self.training_opponent_weapon,
                "agent" if "agent" in self.training_selection_step else "opponent"
            )
        elif self.training_selection_step == "opponent_type":
            self._draw_opponent_type_selector()
        elif self.training_selection_step == "config":
            self._draw_training_config()
        
        # Resumo das seleções
        summary_y = 510
        pygame.draw.line(screen, GRAY, (100, summary_y), (SCREEN_WIDTH - 100, summary_y), 1)
        
        agent_class = self.available_classes[self.training_class]
        agent_weapon = self.available_weapons[self.training_weapon]
        
        if self.training_multi_opponent:
            opponent_text = "TODOS (rotação a cada 50 eps)"
        else:
            opponent_class = self.available_classes[self.training_opponent_class]
            opponent_weapon = self.available_weapons[self.training_opponent_weapon]
            opponent_text = f"{opponent_class.title()} + {opponent_weapon.title()}"
        
        summary_text = f"Agente: {agent_class.title()} + {agent_weapon.title()}  vs  Oponente: {opponent_text}"
        summary_surf = font.render(summary_text, True, (150, 150, 150))
        summary_rect = summary_surf.get_rect(center=(SCREEN_WIDTH // 2, summary_y + 25))
        screen.blit(summary_surf, summary_rect)
        
        # Visualização prévia
        preview_y = 560
        
        # Agente (esquerda)
        pygame.draw.circle(screen, (100, 255, 100), (SCREEN_WIDTH // 2 - 150, preview_y + 40), 30)
        agent_label = font.render("AGENTE", True, (100, 255, 100))
        screen.blit(agent_label, (SCREEN_WIDTH // 2 - 180, preview_y))
        
        # VS
        vs_surf = font_large.render("VS", True, WHITE)
        vs_rect = vs_surf.get_rect(center=(SCREEN_WIDTH // 2, preview_y + 40))
        screen.blit(vs_surf, vs_rect)
        
        # Oponente (direita)
        if self.training_multi_opponent:
            # Desenhar múltiplos círculos pequenos
            for i in range(5):
                color = [(255, 100, 100), (100, 100, 255), (255, 200, 100), (200, 100, 255), (100, 255, 255)][i]
                pygame.draw.circle(screen, color, (SCREEN_WIDTH // 2 + 120 + i * 20, preview_y + 40), 12)
            opp_label = font.render("MÚLTIPLOS", True, (255, 200, 100))
        else:
            pygame.draw.circle(screen, (255, 100, 100), (SCREEN_WIDTH // 2 + 150, preview_y + 40), 30)
            opp_label = font.render("OPONENTE", True, (255, 100, 100))
        screen.blit(opp_label, (SCREEN_WIDTH // 2 + 100, preview_y))
        
        # Instruções
        instructions = "← → Navegar | ENTER Confirmar | BACKSPACE Voltar | ESC Menu"
        inst_surf = font.render(instructions, True, GRAY)
        inst_rect = inst_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        screen.blit(inst_surf, inst_rect)
    
    def _draw_class_selector(self, selected_idx: int, role: str):
        """Desenha seletor de classe"""
        classes = ClassRegistry.get_all()
        class_list = list(classes.items())
        
        role_color = (100, 255, 100) if role == "agent" else (255, 100, 100)
        role_text = "AGENTE (IA que será treinada)" if role == "agent" else "OPONENTE (IA simples)"
        
        role_surf = font_large.render(role_text, True, role_color)
        role_rect = role_surf.get_rect(center=(SCREEN_WIDTH // 2, 175))
        screen.blit(role_surf, role_rect)
        
        # Cards de classes
        card_width = 180
        card_height = 200
        start_x = (SCREEN_WIDTH - (len(class_list) * (card_width + 15) - 15)) // 2
        
        for i, (class_id, cls) in enumerate(class_list):
            x = start_x + i * (card_width + 15)
            y = 210
            
            # Card
            is_selected = (i == selected_idx)
            card_color = (60, 60, 60) if not is_selected else (80, 120, 80)
            border_color = role_color if is_selected else (100, 100, 100)
            
            card_rect = pygame.Rect(x, y, card_width, card_height)
            pygame.draw.rect(screen, card_color, card_rect, border_radius=10)
            pygame.draw.rect(screen, border_color, card_rect, 3 if is_selected else 1, border_radius=10)
            
            # Círculo representando a classe
            circle_color = (100, 200, 100) if is_selected else (150, 150, 150)
            pygame.draw.circle(screen, circle_color, (x + card_width // 2, y + 50), 25)
            
            # Nome da classe
            name_color = WHITE if is_selected else (180, 180, 180)
            name_surf = font_large.render(cls.display_name, True, name_color)
            name_rect = name_surf.get_rect(center=(x + card_width // 2, y + 95))
            screen.blit(name_surf, name_rect)
            
            # Buscar stats do banco de dados central
            custom_stats = self.config_db.get_class_stats(class_id)
            
            # Stats resumidos (usando valores do banco de dados)
            stats_lines = [
                f"HP: {custom_stats.get('max_health', 100):.0f}",
                f"Speed: {custom_stats.get('speed', 2.5):.1f}",
                f"DMG: {custom_stats.get('damage_multiplier', 1.0):.1f}x"
            ]
            
            for j, stat in enumerate(stats_lines):
                stat_color = (150, 150, 150) if not is_selected else (200, 200, 200)
                stat_surf = font.render(stat, True, stat_color)
                stat_rect = stat_surf.get_rect(center=(x + card_width // 2, y + 125 + j * 20))
                screen.blit(stat_surf, stat_rect)
            
            # Indicador de seleção
            if is_selected:
                arrow_surf = font_large.render("▼", True, role_color)
                arrow_rect = arrow_surf.get_rect(center=(x + card_width // 2, y - 15))
                screen.blit(arrow_surf, arrow_rect)
    
    def _draw_weapon_selector(self, selected_idx: int, role: str):
        """Desenha seletor de arma"""
        weapons = WeaponRegistry.get_all()
        weapon_list = list(weapons.items())
        
        role_color = (100, 255, 100) if role == "agent" else (255, 100, 100)
        role_text = "Arma do AGENTE" if role == "agent" else "Arma do OPONENTE"
        
        role_surf = font_large.render(role_text, True, role_color)
        role_rect = role_surf.get_rect(center=(SCREEN_WIDTH // 2, 175))
        screen.blit(role_surf, role_rect)
        
        # Cards de armas
        card_width = 220
        card_height = 200
        start_x = (SCREEN_WIDTH - (len(weapon_list) * (card_width + 15) - 15)) // 2
        
        for i, (weapon_id, weapon_cls) in enumerate(weapon_list):
            x = start_x + i * (card_width + 15)
            y = 210
            
            is_selected = (i == selected_idx)
            card_color = (60, 60, 60) if not is_selected else (80, 100, 120)
            border_color = role_color if is_selected else (100, 100, 100)
            
            card_rect = pygame.Rect(x, y, card_width, card_height)
            pygame.draw.rect(screen, card_color, card_rect, border_radius=10)
            pygame.draw.rect(screen, border_color, card_rect, 3 if is_selected else 1, border_radius=10)
            
            # Ícone de arma (linha representando)
            line_color = (255, 200, 100) if is_selected else (150, 150, 150)
            pygame.draw.line(screen, line_color, (x + 60, y + 50), (x + card_width - 60, y + 50), 4)
            
            # Nome
            name_color = WHITE if is_selected else (180, 180, 180)
            name_surf = font_large.render(weapon_cls.display_name, True, name_color)
            name_rect = name_surf.get_rect(center=(x + card_width // 2, y + 85))
            screen.blit(name_surf, name_rect)
            
            # Buscar stats do banco de dados central
            custom_stats = self.config_db.get_weapon_stats(weapon_id)
            
            stats_lines = [
                f"Dano: {custom_stats.get('base_damage', 20):.0f}",
                f"Alcance: {custom_stats.get('range', 60):.0f}",
                f"Velocidade: {1/custom_stats.get('attack_cooldown', 0.5):.1f}/s"
            ]
            
            for j, stat in enumerate(stats_lines):
                stat_color = (150, 150, 150) if not is_selected else (200, 200, 200)
                stat_surf = font.render(stat, True, stat_color)
                stat_rect = stat_surf.get_rect(center=(x + card_width // 2, y + 115 + j * 22))
                screen.blit(stat_surf, stat_rect)
            
            if is_selected:
                arrow_surf = font_large.render("▼", True, role_color)
                arrow_rect = arrow_surf.get_rect(center=(x + card_width // 2, y - 15))
                screen.blit(arrow_surf, arrow_rect)
    
    def _draw_opponent_type_selector(self):
        """Desenha seletor de tipo de oponente (único ou múltiplos)"""
        title_surf = font_large.render("Escolha o Tipo de Oponente", True, (255, 200, 100))
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, 175))
        screen.blit(title_surf, title_rect)
        
        # Duas opções
        options = [
            {
                "title": "🎯 Oponente Único",
                "desc": "Treina contra uma combinação específica",
                "detail": "Bom para especialização",
                "selected": not self.training_multi_opponent
            },
            {
                "title": "🌟 Múltiplos Oponentes",
                "desc": "Treina contra TODAS as combinações",
                "detail": f"Rotação a cada {self.episodes_per_opponent} episódios",
                "selected": self.training_multi_opponent
            }
        ]
        
        card_width = 380
        card_height = 220
        spacing = 50
        start_x = (SCREEN_WIDTH - (2 * card_width + spacing)) // 2
        
        for i, opt in enumerate(options):
            x = start_x + i * (card_width + spacing)
            y = 220
            
            is_selected = opt["selected"]
            card_color = (60, 80, 60) if is_selected else (45, 45, 45)
            border_color = (100, 255, 100) if is_selected else (80, 80, 80)
            
            card_rect = pygame.Rect(x, y, card_width, card_height)
            pygame.draw.rect(screen, card_color, card_rect, border_radius=15)
            pygame.draw.rect(screen, border_color, card_rect, 4 if is_selected else 2, border_radius=15)
            
            # Título
            title_color = (100, 255, 100) if is_selected else (180, 180, 180)
            title_surf = font_large.render(opt["title"], True, title_color)
            title_rect = title_surf.get_rect(center=(x + card_width // 2, y + 50))
            screen.blit(title_surf, title_rect)
            
            # Descrição
            desc_color = WHITE if is_selected else (150, 150, 150)
            desc_surf = font.render(opt["desc"], True, desc_color)
            desc_rect = desc_surf.get_rect(center=(x + card_width // 2, y + 100))
            screen.blit(desc_surf, desc_rect)
            
            # Detalhe
            detail_color = (100, 200, 100) if is_selected else (100, 100, 100)
            detail_surf = font.render(opt["detail"], True, detail_color)
            detail_rect = detail_surf.get_rect(center=(x + card_width // 2, y + 140))
            screen.blit(detail_surf, detail_rect)
            
            # Indicador de selecionado
            if is_selected:
                check_surf = font_large.render("✓", True, (100, 255, 100))
                screen.blit(check_surf, (x + card_width - 50, y + 15))
                
                # Seta
                arrow_surf = font_large.render("▼", True, (100, 255, 100))
                arrow_rect = arrow_surf.get_rect(center=(x + card_width // 2, y - 15))
                screen.blit(arrow_surf, arrow_rect)
        
        # Info extra para múltiplos oponentes
        if self.training_multi_opponent:
            total_combos = len(self.available_classes) * len(self.available_weapons)
            info_text = f"Total de combinações: {total_combos} ({len(self.available_classes)} classes × {len(self.available_weapons)} armas)"
            info_surf = font.render(info_text, True, (255, 200, 100))
            info_rect = info_surf.get_rect(center=(SCREEN_WIDTH // 2, 465))
            screen.blit(info_surf, info_rect)
    
    def _draw_training_config(self):
        """Desenha configurações de treinamento"""
        title_surf = font_large.render("Configurações de Treinamento", True, (100, 200, 255))
        title_rect = title_surf.get_rect(center=(SCREEN_WIDTH // 2, 175))
        screen.blit(title_surf, title_rect)
        
        # Box de configuração
        config_rect = pygame.Rect(200, 210, SCREEN_WIDTH - 400, 250)
        pygame.draw.rect(screen, (45, 45, 45), config_rect, border_radius=10)
        
        configs = [
            ("Episódios de Treinamento:", f"{self.training_episodes:,}", "↑/↓ para ajustar"),
            ("Velocidade:", f"{self.training_speed}x" if self.training_speed > 0 else "MÁXIMA", "Ajustável durante treino"),
            ("Visualização:", "Ativada" if self.training_render else "Desativada", "V para alternar durante treino"),
            ("Auto-save a cada:", f"{self.training_save_freq:,} episódios", "S para salvar manual")
        ]
        
        for i, (label, value, hint) in enumerate(configs):
            y = 240 + i * 55
            
            # Label
            label_surf = font.render(label, True, (200, 200, 200))
            screen.blit(label_surf, (230, y))
            
            # Valor
            value_surf = font_large.render(value, True, (100, 255, 150))
            screen.blit(value_surf, (500, y - 5))
            
            # Hint
            hint_surf = font.render(hint, True, (100, 100, 100))
            screen.blit(hint_surf, (230, y + 22))
        
        # Botão de iniciar
        start_rect = pygame.Rect(SCREEN_WIDTH // 2 - 150, 420, 300, 50)
        pygame.draw.rect(screen, (80, 150, 80), start_rect, border_radius=25)
        pygame.draw.rect(screen, (100, 200, 100), start_rect, 3, border_radius=25)
        
        start_text = font_large.render("▶ INICIAR TREINAMENTO", True, WHITE)
        start_rect_text = start_text.get_rect(center=start_rect.center)
        screen.blit(start_text, start_rect_text)
        
        hint = font.render("Pressione ENTER para iniciar", True, GRAY)
        hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, 490))
        screen.blit(hint, hint_rect)
    
    def draw_training(self):
        """Desenha a tela de treinamento ativo"""
        # Calcular tamanhos responsivos
        panel_width = min(380, int(SCREEN_WIDTH * 0.3))
        panel_margin = 10
        line_height = max(20, int(SCREEN_HEIGHT * 0.028))
        small_line = max(18, int(SCREEN_HEIGHT * 0.024))
        
        # Transparência dos painéis (0-255, quanto menor mais transparente)
        panel_alpha = 160  # 60% opaco
        
        # Desenhar arena e entidades se visualização ativa
        if self.training_render:
            # Arena
            arena_rect = pygame.Rect(*self.arena_config.playable_rect)
            pygame.draw.rect(screen, (40, 40, 40), arena_rect)
            pygame.draw.rect(screen, (100, 100, 100), arena_rect, 2)
            
            # Entidades (treinamento sempre tem IA)
            for entity in self.entities:
                if entity:
                    # Verificar se é um Assassino para passar o parâmetro against_ai
                    if hasattr(entity, 'invisible'):
                        entity.draw(screen, against_ai=True)
                    else:
                        entity.draw(screen)
            
            # Efeito de slow motion
            if self.slow_motion:
                pulse = int(abs(math.sin(pygame.time.get_ticks() / 100)) * 100) + 100
                border_color = (pulse, pulse // 2, 0)
                pygame.draw.rect(screen, border_color, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 8)
                
                slow_text = font_large.render("⚡ SLOW MOTION ⚡", True, (255, 200, 100))
                slow_rect = slow_text.get_rect(center=(SCREEN_WIDTH // 2, 60))
                screen.blit(slow_text, slow_rect)
        else:
            wait_surf = font_large.render("Treinamento em progresso...", True, WHITE)
            wait_rect = wait_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
            screen.blit(wait_surf, wait_rect)
            
            hint = font.render("Pressione V para visualizar", True, GRAY)
            hint_rect = hint.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            screen.blit(hint, hint_rect)
        
        # ========== PAINEL DE ESTATÍSTICAS (Esquerda) - TRANSPARENTE ==========
        stats = self.training_stats
        episode = stats['episode']
        total = stats['total_episodes']
        wins = stats['wins']
        losses = stats['losses']
        avg_reward = stats['avg_reward']
        best = stats['best_reward']
        
        # Calcular altura do painel baseado no conteúdo
        has_nn = TORCH_AVAILABLE and self.neural_network
        panel_height = 200 + (80 if has_nn else 0)
        
        # Criar surface transparente para o painel
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel_surface.fill((25, 25, 30, panel_alpha))
        screen.blit(panel_surface, (panel_margin, panel_margin))
        
        # Borda do painel (sem transparência)
        panel_rect = pygame.Rect(panel_margin, panel_margin, panel_width, panel_height)
        pygame.draw.rect(screen, (70, 70, 80, 200), panel_rect, 2, border_radius=10)
        
        y_offset = panel_margin + 10
        x_offset = panel_margin + 15
        
        # Título
        nn_status = "🧠" if has_nn else "⚙️"
        panel_title = font_large.render(f"{nn_status} Estatísticas", True, (100, 255, 150))
        screen.blit(panel_title, (x_offset, y_offset))
        y_offset += line_height + 10
        
        # Progresso
        progress = episode / total if total > 0 else 0
        progress_text = f"Episódio: {episode:,} / {total:,} ({progress*100:.1f}%)"
        prog_surf = font.render(progress_text, True, WHITE)
        screen.blit(prog_surf, (x_offset, y_offset))
        y_offset += small_line + 2
        
        # Barra de progresso
        bar_width = panel_width - 30
        bar_rect = pygame.Rect(x_offset, y_offset, bar_width, 12)
        pygame.draw.rect(screen, (50, 50, 55), bar_rect, border_radius=6)
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            fill_rect = pygame.Rect(x_offset, y_offset, fill_width, 12)
            pygame.draw.rect(screen, (80, 180, 80), fill_rect, border_radius=6)
        y_offset += 20
        
        # Estatísticas de combate
        win_rate = (wins / episode * 100) if episode > 0 else 0
        stats_lines = [
            (f"✓ Vitórias: {wins}", (100, 255, 100)),
            (f"✗ Derrotas: {losses}", (255, 100, 100)),
            (f"📊 Taxa: {win_rate:.1f}%", (200, 200, 200)),
            (f"💰 Reward Médio: {avg_reward:.2f}", (100, 200, 255)),
            (f"🏆 Melhor: {best:.2f}", (255, 200, 100))
        ]
        
        for text, color in stats_lines:
            surf = font.render(text, True, color)
            screen.blit(surf, (x_offset, y_offset))
            y_offset += small_line
        
        # Estatísticas da rede neural
        if has_nn:
            y_offset += 5
            pygame.draw.line(screen, (60, 60, 70), (x_offset, y_offset), (x_offset + bar_width, y_offset), 1)
            y_offset += 8
            
            nn_title = font.render("🧠 Rede Neural:", True, (255, 200, 100))
            screen.blit(nn_title, (x_offset, y_offset))
            y_offset += small_line
            
            policy_loss = stats.get('policy_loss', 0)
            value_loss = stats.get('value_loss', 0)
            updates = stats.get('learning_updates', 0)
            
            nn_stats = [
                (f"   Updates: {updates}", (180, 180, 180)),
                (f"   Policy Loss: {policy_loss:.4f}", (255, 150, 150)),
                (f"   Value Loss: {value_loss:.4f}", (150, 150, 255))
            ]
            
            for text, color in nn_stats:
                surf = font.render(text, True, color)
                screen.blit(surf, (x_offset, y_offset))
                y_offset += small_line - 2
        
        # ========== PAINEL DE CONTROLES (Direita) - TRANSPARENTE ==========
        controls_width = min(280, int(SCREEN_WIDTH * 0.22))
        controls_height = 160
        
        # Criar surface transparente para controles
        controls_surface = pygame.Surface((controls_width, controls_height), pygame.SRCALPHA)
        controls_surface.fill((25, 25, 30, panel_alpha))
        screen.blit(controls_surface, (SCREEN_WIDTH - controls_width - panel_margin, panel_margin))
        
        # Borda do painel de controles
        controls_rect = pygame.Rect(SCREEN_WIDTH - controls_width - panel_margin, panel_margin, controls_width, controls_height)
        pygame.draw.rect(screen, (70, 70, 80, 200), controls_rect, 2, border_radius=10)
        
        ctrl_x = SCREEN_WIDTH - controls_width - panel_margin + 15
        ctrl_y = panel_margin + 10
        
        controls_title = font_large.render("🎮 Controles", True, (100, 200, 255))
        screen.blit(controls_title, (ctrl_x, ctrl_y))
        ctrl_y += line_height + 5
        
        controls = [
            "ESPAÇO - Pausar/Continuar",
            "V - Alternar visualização",
            "+/- - Velocidade",
            "S - Salvar modelo",
            "ESC - Parar"
        ]
        
        for ctrl in controls:
            ctrl_surf = font.render(ctrl, True, (160, 160, 160))
            screen.blit(ctrl_surf, (ctrl_x, ctrl_y))
            ctrl_y += small_line
        
        # ========== STATUS BAR (Inferior) - TRANSPARENTE ==========
        # Fundo do status transparente
        status_bar_height = 70
        status_surface = pygame.Surface((SCREEN_WIDTH, status_bar_height), pygame.SRCALPHA)
        status_surface.fill((20, 20, 25, panel_alpha))
        screen.blit(status_surface, (0, SCREEN_HEIGHT - status_bar_height))
        pygame.draw.line(screen, (60, 60, 70), (0, SCREEN_HEIGHT - status_bar_height), (SCREEN_WIDTH, SCREEN_HEIGHT - status_bar_height), 2)
        
        # Status do treinamento
        status_text = "⏸ PAUSADO" if self.training_paused else f"▶ Treinando ({self.training_speed}x)"
        status_color = (255, 200, 100) if self.training_paused else (100, 255, 100)
        status_surf = font_large.render(status_text, True, status_color)
        status_rect = status_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 45))
        screen.blit(status_surf, status_rect)
        
        # Info do combate atual
        if self.training_render and self.training_entity and self.training_opponent:
            agent_hp = self.training_entity.health if self.training_entity.is_alive() else 0
            opp_hp = self.training_opponent.health if self.training_opponent.is_alive() else 0
            
            battle_text = f"🟢 Agente: {agent_hp:.0f} HP  |  🔴 Oponente: {opp_hp:.0f} HP"
            battle_surf = font.render(battle_text, True, WHITE)
            battle_rect = battle_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 18))
            screen.blit(battle_surf, battle_rect)
        
        # Mostrar oponente atual se em modo multi-oponente
        if self.training_multi_opponent and self.training_opponent_pool:
            opp_class, opp_weapon = self._get_current_opponent_config()
            opp_idx = self.current_opponent_idx + 1
            total_opp = len(self.training_opponent_pool)
            eps_until_rotate = self.episodes_per_opponent - (self.training_stats['episode'] % self.episodes_per_opponent)
            
            multi_text = f"🎯 Oponente {opp_idx}/{total_opp}: {opp_class.title()} + {opp_weapon.title()} (troca em {eps_until_rotate} eps)"
            multi_surf = font.render(multi_text, True, (255, 200, 100))
            multi_rect = multi_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - status_bar_height - 15))
            screen.blit(multi_surf, multi_rect)
        
        # Mensagem de salvamento
        if self.save_message_timer > 0:
            self.save_message_timer -= 1/60
            
            # Box de mensagem
            msg_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2 - 30, 400, 60)
            pygame.draw.rect(screen, (30, 80, 30), msg_rect, border_radius=10)
            pygame.draw.rect(screen, (100, 255, 100), msg_rect, 3, border_radius=10)
            
            save_surf = font_large.render("✅ Modelo Salvo!", True, (100, 255, 100))
            save_rect = save_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 10))
            screen.blit(save_surf, save_rect)
            
            file_surf = font.render(self.save_message, True, (200, 200, 200))
            file_rect = file_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 15))
            screen.blit(file_surf, file_rect)

    def run(self):
        """Loop principal do jogo"""
        while self.running:
            dt = clock.tick(FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        pygame.quit()
        sys.exit()


def main():
    """Função principal"""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
