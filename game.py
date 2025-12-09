"""
Circle Warriors - Weapon Ball Fight
====================================
Jogo 2D de combate com círculos armados.
Suporte para múltiplos agentes controlados por IA ou jogadores.

FLUXO DO JOGO:
1. Menu Principal -> Escolher entre TREINO ou PARTIDA
2. Escolher tamanho: 1v1, 2v2, 3v3, 5v5
3. Escolher mapa (pequeno ou grande, todos com fog of war)
4. Configurar cada agente: classe, arma, controlador (IA/modelo/jogador)
5. Iniciar simulação
"""

import pygame
import math
import sys
import os
import random
import json
import time
import numpy as np
from typing import List, Optional, Dict, Tuple
from collections import deque
from dataclasses import dataclass, field

# Imports do jogo
from entities import Entity, ClassRegistry
from weapons import WeaponRegistry
from physics import Physics
from controller import PlayerController, StrategicAI, AIController
from game_state import GameState, ArenaConfig, GameConfig
from fog_of_war import FogOfWar, ObstacleManager, get_vision_radius
from map_configs import (
    MapConfig, ALL_MAPS, SMALL_MAPS, LARGE_MAPS, 
    get_map, list_maps, get_spawn_positions
)

# Tentar importar PyTorch para redes neurais
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
    print("✅ PyTorch disponível - Treinamento com Rede Neural ATIVO")
except ImportError:
    print("⚠️ PyTorch não encontrado - Usando IA estratégica")

# Inicialização do Pygame
pygame.init()

# Configurações da tela
display_info = pygame.display.Info()
SCREEN_WIDTH = int(display_info.current_w * 0.9)
SCREEN_HEIGHT = int(display_info.current_h * 0.85)
SCREEN_WIDTH = max(SCREEN_WIDTH, 1024)
SCREEN_HEIGHT = max(SCREEN_HEIGHT, 700)

os.environ['SDL_VIDEO_CENTERED'] = '1'
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
pygame.display.set_caption("Circle Warriors - Weapon Ball Fight")

# Cores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (30, 30, 30)
BLUE = (100, 150, 255)
RED = (255, 100, 100)
GREEN = (100, 255, 100)
YELLOW = (255, 255, 100)
PURPLE = (200, 100, 255)

# Clock e FPS
clock = pygame.time.Clock()
FPS = 60

# Fontes
def get_fonts():
    base = min(SCREEN_WIDTH, SCREEN_HEIGHT)
    return {
        'small': pygame.font.Font(None, max(18, int(base * 0.022))),
        'medium': pygame.font.Font(None, max(28, int(base * 0.035))),
        'large': pygame.font.Font(None, max(40, int(base * 0.05))),
        'title': pygame.font.Font(None, max(60, int(base * 0.07))),
    }

fonts = get_fonts()


# ============================================================================
# CONFIGURAÇÃO DE AGENTE
# ============================================================================

@dataclass
class AgentConfig:
    """Configuração de um agente no jogo"""
    class_id: str = "warrior"
    weapon_id: str = "sword"
    controller_type: str = "ai"  # "player1", "player2", "ai", "model"
    model_path: Optional[str] = None
    team: str = "blue"  # "blue" ou "red"
    
    def get_display_name(self) -> str:
        cls = ClassRegistry.get_all().get(self.class_id)
        weapon = WeaponRegistry.get_all().get(self.weapon_id)
        cls_name = cls.display_name if cls else self.class_id
        weapon_name = weapon.display_name if weapon else self.weapon_id
        return f"{cls_name} + {weapon_name}"


# ============================================================================
# REDE NEURAL (se PyTorch disponível)
# ============================================================================

if TORCH_AVAILABLE:
    import torch.nn.functional as F
    
    class ActorCritic(nn.Module):
        """Rede neural Actor-Critic para PPO"""
        
        def __init__(self, obs_size: int = 19, action_size: int = 4, hidden_size: int = 256):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            self.actor_mean = nn.Linear(hidden_size, action_size)
            self.actor_log_std = nn.Parameter(torch.zeros(action_size))
            self.critic = nn.Linear(hidden_size, 1)
        
        def forward(self, x):
            shared = self.shared(x)
            action_mean = torch.tanh(self.actor_mean(shared))
            action_std = torch.exp(self.actor_log_std)
            value = self.critic(shared)
            return action_mean, action_std, value
        
        def get_action(self, obs: np.ndarray, deterministic: bool = False):
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action_mean, action_std, value = self.forward(obs_tensor)
                if deterministic:
                    action = action_mean
                else:
                    dist = torch.distributions.Normal(action_mean, action_std)
                    action = dist.sample()
                return action.squeeze().numpy(), value.item()
        
        def evaluate_actions(self, obs, actions):
            """Avalia ações para o treinamento PPO"""
            action_mean, action_std, value = self.forward(obs)
            dist = torch.distributions.Normal(action_mean, action_std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return value.squeeze(), log_probs, entropy


# ============================================================================
# CONTROLADOR DE REDE NEURAL
# ============================================================================

class NeuralNetworkController:
    """Controlador que usa uma rede neural treinada para controlar uma entidade"""
    
    def __init__(self, network, game):
        self.network = network
        self.game = game
        self.targets = []
        self.entity = None  # Será setado por entity.set_controller()
    
    def set_targets(self, targets):
        """Define os alvos"""
        self.targets = targets
    
    def update(self, dt: float):
        """Atualiza a entidade usando a rede neural"""
        if not self.entity or not self.entity.is_alive():
            return
        
        # Encontrar alvo mais próximo
        target = self._find_closest_target()
        if not target:
            return
        
        # Obter observação
        obs = self._get_observation(target)
        
        # Obter ação da rede neural
        if TORCH_AVAILABLE and self.network:
            action, _ = self.network.get_action(obs, deterministic=True)
        else:
            action = np.zeros(4)
        
        # Aplicar ação
        self._apply_action(action)
    
    def _find_closest_target(self):
        """Encontra o alvo vivo mais próximo"""
        closest = None
        min_dist = float('inf')
        
        for target in self.targets:
            if target.is_alive():
                dx = target.x - self.entity.x
                dy = target.y - self.entity.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
                    closest = target
        
        return closest
    
    def _get_observation(self, target) -> np.ndarray:
        """Retorna observação normalizada"""
        map_w = self.game.current_map.width if self.game.current_map else SCREEN_WIDTH
        map_h = self.game.current_map.height if self.game.current_map else SCREEN_HEIGHT
        
        entity = self.entity
        entity_stats = entity.stats_manager.get_stats()
        target_stats = target.stats_manager.get_stats()
        
        obs = []
        
        # Estado do agente (9 valores)
        obs.extend([
            entity.x / map_w,
            entity.y / map_h,
            entity.vx / 20,
            entity.vy / 20,
            entity.facing_angle / math.pi,
            entity.health / entity_stats.max_health,
            1.0 if entity.invulnerable_time > 0 else 0.0,
            1.0 if entity.weapon and entity.weapon.can_attack else 0.0,
            1.0 if entity.ability_cooldown <= 0 else 0.0
        ])
        
        # Estado do alvo relativo (8 valores)
        rel_x = (target.x - entity.x) / map_w
        rel_y = (target.y - entity.y) / map_h
        distance = math.sqrt(rel_x**2 + rel_y**2)
        angle_to = math.atan2(rel_y, rel_x)
        
        obs.extend([
            rel_x,
            rel_y,
            distance,
            angle_to / math.pi,
            target.vx / 20,
            target.vy / 20,
            target.health / target_stats.max_health,
            1.0 if target.weapon and target.weapon.is_attacking else 0.0
        ])
        
        # Distância das bordas (2 valores)
        border_x = min(entity.x, map_w - entity.x) / (map_w / 2)
        border_y = min(entity.y, map_h - entity.y) / (map_h / 2)
        obs.extend([border_x, border_y])
        
        return np.array(obs, dtype=np.float32)
    
    def _apply_action(self, action: np.ndarray):
        """Aplica ação à entidade"""
        entity = self.entity
        
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


# ============================================================================
# CLASSE PRINCIPAL DO JOGO
# ============================================================================

class Game:
    """Classe principal do jogo com novo sistema de menus"""
    
    def __init__(self):
        # Estado do jogo
        self.running = True
        self.paused = False
        self.game_over = False
        self.winner_team: Optional[str] = None
        
        # Física e entidades
        self.physics = Physics(SCREEN_WIDTH, SCREEN_HEIGHT)
        self.entities: List[Entity] = []
        self.blue_team: List[Entity] = []
        self.red_team: List[Entity] = []
        
        # ==========================================
        # NOVO SISTEMA DE MENUS
        # ==========================================
        
        # Estados de menu possíveis:
        # "main_menu" -> "mode_select" -> "size_select" -> "map_select" -> "agent_config" -> "playing"/"training"
        self.menu_state = "main_menu"
        
        # Seleções do fluxo
        self.selected_mode = "match"  # "match" ou "training"
        self.selected_size = 1  # 1, 2, 3 ou 5 (para 1v1, 2v2, 3v3, 5v5)
        self.selected_map_idx = 0
        self.selected_map: Optional[MapConfig] = None
        
        # Configuração dos agentes (preenchida na tela de agent_config)
        self.blue_agents: List[AgentConfig] = []
        self.red_agents: List[AgentConfig] = []
        
        # Cursor de navegação nos menus
        self.menu_cursor = 0
        self.agent_config_cursor = 0  # Qual agente está sendo configurado
        self.agent_config_field = 0   # Qual campo do agente (0=classe, 1=arma, 2=controlador)
        self.configuring_team = "blue"  # Qual time está sendo configurado
        
        # Listas de opções
        self.available_classes = ClassRegistry.list_classes()
        self.available_weapons = WeaponRegistry.list_weapons()
        self.available_sizes = [1, 2, 3, 5]
        self.available_models = self._scan_models()
        
        # ==========================================
        # SISTEMA DE MAPA E FOG OF WAR
        # ==========================================
        self.current_map: Optional[MapConfig] = None
        self.fog_of_war: Optional[FogOfWar] = None
        self.obstacle_manager: Optional[ObstacleManager] = None
        
        # ==========================================
        # SISTEMA DE TREINAMENTO COMPLETO
        # ==========================================
        self.training_active = False
        self.training_paused = False
        self.training_render = True
        self.training_speed = 1  # 1=normal, 2+=mais rápido, 0=máximo
        
        # Configuração do agente treinando
        self.training_class = 0
        self.training_weapon = 0
        
        # Modo multi-oponente
        self.training_multi_opponent = True  # Treina contra vários tipos
        self.training_opponent_pool = []  # Lista de (classe_idx, arma_idx)
        self.current_opponent_idx = 0
        self.episodes_per_opponent = 50
        
        # Configurações
        self.training_max_episodes = 5000
        self.training_save_freq = 500
        
        # Estado do episódio
        self.training_entity = None
        self.training_opponent = None
        self.training_step = 0
        self.training_episode_reward = 0
        self._prev_agent_health = 100
        self._prev_opponent_health = 100
        self._tried_attack_this_frame = False
        
        # Estatísticas
        self.training_stats = {
            'episode': 0,
            'total_episodes': 5000,
            'wins': 0,
            'losses': 0,
            'avg_reward': 0,
            'recent_rewards': deque(maxlen=100),
            'best_reward': float('-inf'),
            'training_time': 0,
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'learning_updates': 0
        }
        
        # Rede neural
        self.neural_network = None
        self.nn_optimizer = None
        self.nn_obs_size = 19
        self.nn_action_size = 4
        
        # Buffer de experiência PPO
        self.experience_buffer = {
            'obs': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': []
        }
        
        # Hiperparâmetros PPO
        self.ppo_gamma = 0.99
        self.ppo_gae_lambda = 0.95
        self.ppo_clip_epsilon = 0.2
        self.ppo_epochs = 10
        self.ppo_batch_size = 64
        self.ppo_update_freq = 2048
        self.training_step_count = 0
        
        # ==========================================
        # SELEÇÃO DE MODELO PARA PARTIDA
        # ==========================================
        self.model_select_active = False  # Se está selecionando modelo
        self.model_select_agent_idx = 0   # Qual agente está selecionando modelo
        self.model_select_team = "blue"   # Qual time
        self.model_select_cursor = 0      # Cursor na lista de modelos
        
        # ==========================================
        # MENU DE GERENCIAMENTO DE MODELOS
        # ==========================================
        self.training_menu_cursor = 0
        self.training_menu_state = "main"  # main, new_model, view_model, confirm_delete
        self.new_model_class = 0
        self.new_model_weapon = 0
        self.new_model_episodes = 5000
        self.new_model_map = 0  # Índice do mapa para treinamento
        self.new_model_field = 0  # 0=classe, 1=arma, 2=mapa, 3=episódios, 4=iniciar
        self.viewing_model_idx = 0
        self.delete_confirm = False
        self.training_map_list = list(ALL_MAPS.keys())  # Lista de mapas disponíveis

        # ==========================================
        # SISTEMA DE CÂMERA SIMPLES
        # ==========================================
        self.cam_x = 0
        self.cam_y = 0
        self.cam_zoom = 1.0
        self.cam_target_mode = "p1"  # p1, p2, center
        
        # Histórico para gráficos (últimos 500 episódios)
        self.reward_history = deque(maxlen=500)
        self.win_rate_history = deque(maxlen=100)
        
        # ==========================================
        # EFEITOS VISUAIS
        # ==========================================
        self.slow_motion = False
        self.slow_motion_timer = 0
        self.slow_motion_duration = 0.3
        self.slow_motion_scale = 0.15
        
        # Mensagens de feedback
        self.message = ""
        self.message_timer = 0
    
    def _scan_models(self) -> List[Dict]:
        """Escaneia modelos de IA disponíveis na pasta models/"""
        models = []
        models_dir = "models"
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.json'):
                    path = os.path.join(models_dir, f)
                    try:
                        with open(path, 'r') as file:
                            data = json.load(file)
                            models.append({
                                'filename': f,
                                'path': path,
                                'agent_class': data.get('agent_class', 'unknown'),
                                'agent_weapon': data.get('agent_weapon', 'unknown'),
                                'episode': data.get('episode', 0),
                                'total_episodes': data.get('total_episodes', 0),
                                'wins': data.get('wins', 0),
                                'losses': data.get('losses', 0),
                                'win_rate': data.get('win_rate', 0),
                                'avg_reward': data.get('avg_reward', 0),
                                'best_reward': data.get('best_reward', 0),
                                'training_time': data.get('training_time', 0),
                                'has_neural_weights': data.get('has_neural_weights', False),
                                'weights_file': data.get('weights_file', None),
                                'multi_opponent': data.get('multi_opponent', False),
                            })
                    except:
                        pass
        # Ordenar por episódio (mais treinados primeiro)
        models.sort(key=lambda x: x.get('episode', 0), reverse=True)
        return models
        return models
    
    # ==========================================================================
    # CRIAÇÃO DE ENTIDADES
    # ==========================================================================
    
    def create_entity(self, config: AgentConfig, x: float, y: float, 
                      color: Tuple[int, int, int]) -> Entity:
        """Cria uma entidade a partir de uma configuração de agente"""
        entity = ClassRegistry.create(config.class_id, x, y, color)
        entity.set_weapon(config.weapon_id)
        entity.team = config.team
        
        # Configurar controlador
        controller = self._create_controller(config, entity)
        if controller:
            entity.set_controller(controller)
        
        entity.game_entities = self.entities
        self.entities.append(entity)
        
        if config.team == "blue":
            self.blue_team.append(entity)
        else:
            self.red_team.append(entity)
        
        return entity
    
    def _create_controller(self, config: AgentConfig, entity: Entity):
        """Cria o controlador apropriado para um agente"""
        if config.controller_type == "player1":
            return PlayerController(
                pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d,
                pygame.K_SPACE, pygame.K_LSHIFT
            )
        elif config.controller_type == "player2":
            return PlayerController(
                pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT,
                pygame.K_RETURN, pygame.K_RSHIFT
            )
        elif config.controller_type == "ai":
            ai = StrategicAI(
                class_name=config.class_id.capitalize(),
                weapon_name=config.weapon_id.capitalize()
            )
            return ai
        elif config.controller_type == "model" and config.model_path:
            # Tentar carregar modelo de IA treinada
            return self._load_model_controller(config.model_path)
        
        # Default: IA estratégica
        return StrategicAI(
            class_name=config.class_id.capitalize(),
            weapon_name=config.weapon_id.capitalize()
        )
    
    def _load_model_controller(self, model_path: str):
        """Carrega um controlador de IA a partir de um modelo salvo"""
        if not TORCH_AVAILABLE:
            print("⚠️ PyTorch não disponível, usando IA estratégica")
            return StrategicAI()
        
        # Carregar info do modelo
        try:
            with open(model_path, 'r') as f:
                model_info = json.load(f)
            
            weights_file = model_info.get('weights_file')
            if not weights_file:
                print("⚠️ Modelo sem pesos de rede neural")
                return StrategicAI()
            
            weights_path = os.path.join(os.path.dirname(model_path), weights_file)
            if not os.path.exists(weights_path):
                print(f"⚠️ Arquivo de pesos não encontrado: {weights_path}")
                return StrategicAI()
            
            # Carregar rede neural
            checkpoint = torch.load(weights_path, map_location='cpu')
            obs_size = checkpoint.get('obs_size', 19)
            action_size = checkpoint.get('action_size', 4)
            
            network = ActorCritic(obs_size=obs_size, action_size=action_size, hidden_size=256)
            network.load_state_dict(checkpoint['model_state_dict'])
            network.eval()
            
            # Criar controlador de rede neural
            return NeuralNetworkController(network, self)
        
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo: {e}")
            return StrategicAI()
    
    # ==========================================================================
    # CONFIGURAÇÃO DO JOGO
    # ==========================================================================
    
    def setup_game(self):
        """Configura o jogo com base nas seleções feitas"""
        self.entities.clear()
        self.blue_team.clear()
        self.red_team.clear()
        self.game_over = False
        self.winner_team = None
        
        if not self.selected_map:
            return
        
        map_cfg = self.selected_map
        self.current_map = map_cfg  # Guardar referência ao mapa atual
        
        # Configurar sistemas de mapa
        self._setup_map_systems(map_cfg)
        
        # Obter posições de spawn
        spawn_positions = get_spawn_positions(
            map_cfg, 
            num_teams=2, 
            team_size=self.selected_size
        )
        
        # Criar agentes do time azul
        for i, agent_cfg in enumerate(self.blue_agents):
            if i < len(spawn_positions[0]):
                x, y = spawn_positions[0][i]
                entity = self.create_entity(agent_cfg, x, y, BLUE)
        
        # Criar agentes do time vermelho
        for i, agent_cfg in enumerate(self.red_agents):
            if i < len(spawn_positions[1]):
                x, y = spawn_positions[1][i]
                entity = self.create_entity(agent_cfg, x, y, RED)
        
        # Configurar alvos das IAs
        self._setup_ai_targets()
        
        # Resetar câmera
        self.cam_target_mode = "p1"
        self.cam_zoom = 1.0
        self.cam_x = 0
        self.cam_y = 0
    
    def _setup_map_systems(self, map_cfg: MapConfig):
        """Configura fog of war, obstáculos e câmera para o mapa"""
        # Fog of War (sempre ativo)
        self.fog_of_war = FogOfWar(map_cfg.width, map_cfg.height)
        
        # Obstáculos
        self.obstacle_manager = ObstacleManager()
        self.obstacle_manager.generate_for_map(
            map_width=map_cfg.width,
            map_height=map_cfg.height,
            map_type=map_cfg.obstacle_type,
            border=100,
            density=map_cfg.obstacle_density
        )
        self.fog_of_war.set_obstacle_manager(self.obstacle_manager)
        
        # Atualizar física
        self.physics = Physics(map_cfg.width, map_cfg.height)

    def _update_camera(self):
        """Atualiza posição da câmera (Simples e Robusto)"""
        if not self.current_map:
            return

        target = None
        
        # Determinar alvo baseado no modo
        if self.menu_state == "training":
            target = self.training_entity
        else:
            # Modo de jogo normal
            if self.cam_target_mode == "p1":
                # Tenta pegar jogador humano ou primeiro do time azul
                for entity in self.blue_team:
                    if hasattr(entity, 'controller') and isinstance(entity.controller, PlayerController):
                        target = entity
                        break
                if not target and self.blue_team:
                    target = self.blue_team[0]
            elif self.cam_target_mode == "p2":
                # Primeiro do time vermelho
                if self.red_team:
                    target = self.red_team[0]
            elif self.cam_target_mode == "center":
                # Centro do mapa
                self.cam_x = (self.current_map.width - SCREEN_WIDTH / self.cam_zoom) / 2
                self.cam_y = (self.current_map.height - SCREEN_HEIGHT / self.cam_zoom) / 2
                return # Já definiu posição, sai

        # Se tem alvo, centralizar nele
        if target:
            self.cam_x = target.x - (SCREEN_WIDTH / 2) / self.cam_zoom
            self.cam_y = target.y - (SCREEN_HEIGHT / 2) / self.cam_zoom
        
        # Limitar aos limites do mapa (Clamp)
        view_w = SCREEN_WIDTH / self.cam_zoom
        view_h = SCREEN_HEIGHT / self.cam_zoom
        
        self.cam_x = max(0, min(self.cam_x, self.current_map.width - view_w))
        self.cam_y = max(0, min(self.cam_y, self.current_map.height - view_h))

    def _setup_ai_targets(self):
        """Configura os alvos das IAs"""
        for entity in self.blue_team:
            if hasattr(entity, 'controller') and hasattr(entity.controller, 'set_targets'):
                entity.controller.set_targets(self.red_team)
        
        for entity in self.red_team:
            if hasattr(entity, 'controller') and hasattr(entity.controller, 'set_targets'):
                entity.controller.set_targets(self.blue_team)
    
    # ==========================================================================
    # HANDLERS DE INPUT
    # ==========================================================================
    
    def handle_events(self):
        """Processa eventos do pygame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key)
            
            elif event.type == pygame.VIDEORESIZE:
                global SCREEN_WIDTH, SCREEN_HEIGHT, screen, fonts
                SCREEN_WIDTH = max(1024, event.w)
                SCREEN_HEIGHT = max(700, event.h)
                screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
                fonts = get_fonts()
    
    def _handle_keydown(self, key):
        """Processa tecla pressionada baseado no estado atual"""
        if key == pygame.K_ESCAPE:
            self._handle_escape()
            return
        
        if self.menu_state == "main_menu":
            self._handle_main_menu_input(key)
        elif self.menu_state == "mode_select":
            self._handle_mode_select_input(key)
        elif self.menu_state == "size_select":
            self._handle_size_select_input(key)
        elif self.menu_state == "map_select":
            self._handle_map_select_input(key)
        elif self.menu_state == "agent_config":
            self._handle_agent_config_input(key)
        elif self.menu_state == "playing":
            self._handle_playing_input(key)
        elif self.menu_state == "training":
            self._handle_training_input(key)
        elif self.menu_state == "training_menu":
            self._handle_training_menu_input(key)

        # Hotkeys globais de câmera (funcionam em partida e treinamento)
        if self.menu_state in ["playing", "training"]:
            self._handle_camera_hotkeys(key)
    
    def _handle_escape(self):
        """Volta para o estado anterior ou sai"""
        # Se está em submenu do training_menu
        if self.menu_state == "training_menu":
            if self.training_menu_state != "main":
                self.training_menu_state = "main"
                self.delete_confirm = False
                return
            else:
                self.menu_state = "main_menu"
                self.menu_cursor = 0
                return
        
        transitions = {
            "mode_select": "main_menu",
            "size_select": "main_menu",
            "map_select": "size_select",
            "agent_config": "map_select",
            "playing": "main_menu",
            "training": "training_menu",
        }
        if self.menu_state in transitions:
            self.menu_state = transitions[self.menu_state]
            self.menu_cursor = 0
            if self.menu_state == "training_menu":
                self.training_active = False
                self.available_models = self._scan_models()
        elif self.menu_state == "main_menu":
            self.running = False
    
    def _handle_main_menu_input(self, key):
        """Input do menu principal"""
        if key in [pygame.K_UP, pygame.K_w]:
            self.menu_cursor = (self.menu_cursor - 1) % 2
        elif key in [pygame.K_DOWN, pygame.K_s]:
            self.menu_cursor = (self.menu_cursor + 1) % 2
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            if self.menu_cursor == 0:
                # PARTIDA
                self.selected_mode = "match"
                self.menu_state = "size_select"
                self.menu_cursor = 0
            else:
                # TREINO - vai para menu de gerenciamento
                self.menu_state = "training_menu"
                self.training_menu_state = "main"
                self.training_menu_cursor = 0
                self.available_models = self._scan_models()
    
    def _handle_mode_select_input(self, key):
        """Input da seleção de modo (não usado mais, vai direto para size)"""
        pass
    
    def _handle_size_select_input(self, key):
        """Input da seleção de tamanho (1v1, 2v2, etc)"""
        if key in [pygame.K_UP, pygame.K_w]:
            self.menu_cursor = (self.menu_cursor - 1) % len(self.available_sizes)
        elif key in [pygame.K_DOWN, pygame.K_s]:
            self.menu_cursor = (self.menu_cursor + 1) % len(self.available_sizes)
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            self.selected_size = self.available_sizes[self.menu_cursor]
            self.menu_state = "map_select"
            self.menu_cursor = 0
            self.selected_map_idx = 0
    
    def _handle_map_select_input(self, key):
        """Input da seleção de mapa"""
        all_maps = list(ALL_MAPS.values())
        
        if key in [pygame.K_UP, pygame.K_w]:
            self.selected_map_idx = (self.selected_map_idx - 1) % len(all_maps)
        elif key in [pygame.K_DOWN, pygame.K_s]:
            self.selected_map_idx = (self.selected_map_idx + 1) % len(all_maps)
        elif key in [pygame.K_LEFT, pygame.K_a]:
            self.selected_map_idx = (self.selected_map_idx - 1) % len(all_maps)
        elif key in [pygame.K_RIGHT, pygame.K_d]:
            self.selected_map_idx = (self.selected_map_idx + 1) % len(all_maps)
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            self.selected_map = all_maps[self.selected_map_idx]
            self._init_agent_configs()
            self.menu_state = "agent_config"
            self.agent_config_cursor = 0
            self.agent_config_field = 0
            self.configuring_team = "blue"
    
    def _init_agent_configs(self):
        """Inicializa configurações de agentes com defaults"""
        self.blue_agents = []
        self.red_agents = []
        
        for i in range(self.selected_size):
            # Time azul - primeiro agente é player1 se for partida
            ctrl = "player1" if (i == 0 and self.selected_mode == "match") else "ai"
            self.blue_agents.append(AgentConfig(
                class_id="warrior",
                weapon_id="sword", 
                controller_type=ctrl,
                team="blue"
            ))
            
            # Time vermelho - sempre IA
            self.red_agents.append(AgentConfig(
                class_id="warrior",
                weapon_id="sword",
                controller_type="ai",
                team="red"
            ))
    
    def _handle_agent_config_input(self, key):
        """Input da configuração de agentes"""
        # Se está selecionando modelo, usar handler específico
        if self.model_select_active:
            self._handle_model_select_input(key)
            return
        
        # Determinar lista atual
        current_list = self.blue_agents if self.configuring_team == "blue" else self.red_agents
        
        if not current_list:
            return
        
        current_agent = current_list[self.agent_config_cursor]
        
        if key in [pygame.K_UP, pygame.K_w]:
            # Mover para cima (agente anterior ou trocar de time)
            if self.agent_config_cursor > 0:
                self.agent_config_cursor -= 1
            elif self.configuring_team == "red":
                self.configuring_team = "blue"
                self.agent_config_cursor = len(self.blue_agents) - 1
        
        elif key in [pygame.K_DOWN, pygame.K_s]:
            # Mover para baixo (próximo agente ou trocar de time)
            if self.agent_config_cursor < len(current_list) - 1:
                self.agent_config_cursor += 1
            elif self.configuring_team == "blue":
                self.configuring_team = "red"
                self.agent_config_cursor = 0
        
        elif key in [pygame.K_LEFT, pygame.K_a]:
            # Mudar campo para esquerda
            self._cycle_agent_field(current_agent, -1)
        
        elif key in [pygame.K_RIGHT, pygame.K_d]:
            # Mudar campo para direita
            self._cycle_agent_field(current_agent, 1)
        
        elif key == pygame.K_TAB:
            # Alternar entre campos (classe/arma/controlador)
            self.agent_config_field = (self.agent_config_field + 1) % 3
        
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            # Confirmar e iniciar jogo
            self.setup_game()
            if self.selected_mode == "match":
                self.menu_state = "playing"
            else:
                self.menu_state = "training"
                self.training_active = True
    
    def _cycle_agent_field(self, agent: AgentConfig, direction: int):
        """Cicla o valor do campo atual do agente"""
        if self.agent_config_field == 0:  # Classe
            idx = self.available_classes.index(agent.class_id)
            idx = (idx + direction) % len(self.available_classes)
            agent.class_id = self.available_classes[idx]
        
        elif self.agent_config_field == 1:  # Arma
            idx = self.available_weapons.index(agent.weapon_id)
            idx = (idx + direction) % len(self.available_weapons)
            agent.weapon_id = self.available_weapons[idx]
        
        elif self.agent_config_field == 2:  # Controlador
            controllers = ["ai", "player1", "player2"]
            if self.available_models:
                controllers.append("model")
            idx = controllers.index(agent.controller_type) if agent.controller_type in controllers else 0
            idx = (idx + direction) % len(controllers)
            agent.controller_type = controllers[idx]
            
            # Se selecionou model, abrir tela de seleção de modelo
            if agent.controller_type == "model" and self.available_models:
                self.model_select_active = True
                self.model_select_cursor = 0
                self.model_select_team = self.configuring_team
                self.model_select_agent_idx = self.agent_config_cursor
    
    def _handle_playing_input(self, key):
        """Input durante o jogo"""
        if key == pygame.K_p:
            self.paused = not self.paused
        elif key == pygame.K_r and self.game_over:
            self.setup_game()
    
    def _handle_training_input(self, key):
        """Input durante treinamento"""
        if key == pygame.K_p or key == pygame.K_SPACE:
            self.training_paused = not self.training_paused
        elif key == pygame.K_v:
            self.training_render = not self.training_render
        elif key == pygame.K_PLUS or key == pygame.K_EQUALS:
            self.training_speed = min(10, self.training_speed + 1)
        elif key == pygame.K_MINUS:
            self.training_speed = max(0, self.training_speed - 1)
        elif key == pygame.K_s:
            self._save_training_model()

    def _handle_camera_hotkeys(self, key):
        """Atalhos de câmera simplificados"""
        # Alternar alvo da câmera
        if key == pygame.K_c:
            modes = ["p1", "p2", "center"]
            try:
                current_idx = modes.index(self.cam_target_mode)
                next_idx = (current_idx + 1) % len(modes)
                self.cam_target_mode = modes[next_idx]
            except ValueError:
                self.cam_target_mode = "center"
                
        # Zoom
        elif key in [pygame.K_LEFTBRACKET, pygame.K_z, pygame.K_KP_MINUS, pygame.K_MINUS]:
            self.cam_zoom = max(0.1, self.cam_zoom - 0.1)
        elif key in [pygame.K_RIGHTBRACKET, pygame.K_x, pygame.K_KP_PLUS, pygame.K_EQUALS]:
            self.cam_zoom = min(5.0, self.cam_zoom + 0.1)
        elif key == pygame.K_0 or key == pygame.K_KP0:
            self.cam_zoom = 1.0
    
    def _handle_model_select_input(self, key):
        """Input para seleção de modelo treinado"""
        if key == pygame.K_ESCAPE:
            self.model_select_active = False
            # Voltar para AI se cancelou
            agents = self.blue_agents if self.model_select_team == "blue" else self.red_agents
            if self.model_select_agent_idx < len(agents):
                agents[self.model_select_agent_idx].controller_type = "ai"
            return
        
        if key in [pygame.K_UP, pygame.K_w]:
            self.model_select_cursor = (self.model_select_cursor - 1) % max(1, len(self.available_models))
        elif key in [pygame.K_DOWN, pygame.K_s]:
            self.model_select_cursor = (self.model_select_cursor + 1) % max(1, len(self.available_models))
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            # Confirmar seleção
            if self.available_models and self.model_select_cursor < len(self.available_models):
                model = self.available_models[self.model_select_cursor]
                agents = self.blue_agents if self.model_select_team == "blue" else self.red_agents
                if self.model_select_agent_idx < len(agents):
                    agents[self.model_select_agent_idx].model_path = model['path']
                    # Também atualizar classe e arma para combinar com o modelo
                    if model.get('agent_class') in self.available_classes:
                        agents[self.model_select_agent_idx].class_id = model['agent_class']
                    if model.get('agent_weapon') in self.available_weapons:
                        agents[self.model_select_agent_idx].weapon_id = model['agent_weapon']
            self.model_select_active = False
    
    def _handle_training_menu_input(self, key):
        """Input do menu de gerenciamento de treinamento"""
        if self.training_menu_state == "main":
            self._handle_training_menu_main(key)
        elif self.training_menu_state == "new_model":
            self._handle_training_menu_new_model(key)
        elif self.training_menu_state == "view_model":
            self._handle_training_menu_view_model(key)
        elif self.training_menu_state == "confirm_delete":
            self._handle_training_menu_confirm_delete(key)
    
    def _handle_training_menu_main(self, key):
        """Input do menu principal de treinamento"""
        num_options = 3 + len(self.available_models)  # Novo, Voltar, --- + modelos
        
        if key in [pygame.K_UP, pygame.K_w]:
            self.training_menu_cursor = (self.training_menu_cursor - 1) % num_options
        elif key in [pygame.K_DOWN, pygame.K_s]:
            self.training_menu_cursor = (self.training_menu_cursor + 1) % num_options
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            if self.training_menu_cursor == 0:
                # Novo modelo
                self.training_menu_state = "new_model"
                self.new_model_class = 0
                self.new_model_weapon = 0
                self.new_model_map = 0
                self.new_model_episodes = 5000
                self.new_model_field = 0
            elif self.training_menu_cursor == 1:
                # Voltar
                self.menu_state = "main_menu"
                self.menu_cursor = 0
            elif self.training_menu_cursor >= 3:
                # Ver modelo
                model_idx = self.training_menu_cursor - 3
                if model_idx < len(self.available_models):
                    self.viewing_model_idx = model_idx
                    self.training_menu_state = "view_model"
        elif key == pygame.K_DELETE or key == pygame.K_BACKSPACE:
            # Apagar modelo selecionado
            if self.training_menu_cursor >= 3:
                model_idx = self.training_menu_cursor - 3
                if model_idx < len(self.available_models):
                    self.viewing_model_idx = model_idx
                    self.training_menu_state = "confirm_delete"
                    self.delete_confirm = False
    
    def _handle_training_menu_new_model(self, key):
        """Input da criação de novo modelo"""
        if key in [pygame.K_UP, pygame.K_w]:
            self.new_model_field = (self.new_model_field - 1) % 5
        elif key in [pygame.K_DOWN, pygame.K_s]:
            self.new_model_field = (self.new_model_field + 1) % 5
        elif key in [pygame.K_LEFT, pygame.K_a]:
            if self.new_model_field == 0:  # Classe
                self.new_model_class = (self.new_model_class - 1) % len(self.available_classes)
            elif self.new_model_field == 1:  # Arma
                self.new_model_weapon = (self.new_model_weapon - 1) % len(self.available_weapons)
            elif self.new_model_field == 2:  # Mapa
                self.new_model_map = (self.new_model_map - 1) % len(self.training_map_list)
            elif self.new_model_field == 3:  # Episódios
                self.new_model_episodes = max(100, self.new_model_episodes - 500)
        elif key in [pygame.K_RIGHT, pygame.K_d]:
            if self.new_model_field == 0:  # Classe
                self.new_model_class = (self.new_model_class + 1) % len(self.available_classes)
            elif self.new_model_field == 1:  # Arma
                self.new_model_weapon = (self.new_model_weapon + 1) % len(self.available_weapons)
            elif self.new_model_field == 2:  # Mapa
                self.new_model_map = (self.new_model_map + 1) % len(self.training_map_list)
            elif self.new_model_field == 3:  # Episódios
                self.new_model_episodes = min(50000, self.new_model_episodes + 500)
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            if self.new_model_field == 4:  # Iniciar treinamento
                self._start_new_training()
    
    def _handle_training_menu_view_model(self, key):
        """Input da visualização de modelo"""
        if key in [pygame.K_LEFT, pygame.K_a]:
            self.viewing_model_idx = (self.viewing_model_idx - 1) % max(1, len(self.available_models))
        elif key in [pygame.K_RIGHT, pygame.K_d]:
            self.viewing_model_idx = (self.viewing_model_idx + 1) % max(1, len(self.available_models))
        elif key == pygame.K_DELETE or key == pygame.K_BACKSPACE:
            self.training_menu_state = "confirm_delete"
            self.delete_confirm = False
        elif key == pygame.K_c:
            # Continuar treinamento
            if self.available_models and self.viewing_model_idx < len(self.available_models):
                self._continue_training(self.available_models[self.viewing_model_idx])
    
    def _handle_training_menu_confirm_delete(self, key):
        """Input da confirmação de exclusão"""
        if key in [pygame.K_LEFT, pygame.K_a, pygame.K_RIGHT, pygame.K_d]:
            self.delete_confirm = not self.delete_confirm
        elif key in [pygame.K_RETURN, pygame.K_SPACE]:
            if self.delete_confirm:
                self._delete_model(self.viewing_model_idx)
                self.training_menu_state = "main"
                self.training_menu_cursor = 0
            else:
                self.training_menu_state = "main"
    
    def _start_new_training(self):
        """Inicia treinamento de novo modelo"""
        self.training_class = self.new_model_class
        self.training_weapon = self.new_model_weapon
        self.training_max_episodes = self.new_model_episodes
        
        # Configurar mapa selecionado para treinamento
        map_key = self.training_map_list[self.new_model_map]
        self.selected_map = ALL_MAPS[map_key]
        self.current_map = self.selected_map
        self._setup_map_systems(self.selected_map)
        
        # Configurar câmera para mostrar arena toda
        self.cam_target_mode = "center"
        self.cam_zoom = 1.0
        self._update_camera()
        
        # Limpar históricos
        self.reward_history.clear()
        self.win_rate_history.clear()
        
        # Iniciar treinamento
        self.menu_state = "training"
        self._start_training()  # Iniciar o treinamento real
    
    def _continue_training(self, model: Dict):
        """Continua treinamento de um modelo existente"""
        # Encontrar classe e arma
        try:
            self.training_class = self.available_classes.index(model['agent_class'])
            self.training_weapon = self.available_weapons.index(model['agent_weapon'])
        except ValueError:
            self.training_class = 0
            self.training_weapon = 0
        
        # Carregar rede neural existente se disponível
        if model.get('has_neural_weights') and model.get('weights_file'):
            weights_path = os.path.join("models", model['weights_file'])
            if os.path.exists(weights_path) and TORCH_AVAILABLE:
                try:
                    checkpoint = torch.load(weights_path, map_location='cpu')
                    self.neural_network = ActorCritic(
                        obs_size=checkpoint.get('obs_size', 19),
                        action_size=checkpoint.get('action_size', 4),
                        hidden_size=256
                    )
                    self.neural_network.load_state_dict(checkpoint['model_state_dict'])
                    self.nn_optimizer = torch.optim.Adam(self.neural_network.parameters(), lr=3e-4)
                    if 'optimizer_state_dict' in checkpoint:
                        self.nn_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f"✅ Modelo carregado: {model['filename']}")
                except Exception as e:
                    print(f"⚠️ Erro ao carregar modelo: {e}")
                    self.neural_network = None
        
        # Restaurar estatísticas
        self.training_stats['episode'] = model.get('episode', 0)
        self.training_stats['wins'] = model.get('wins', 0)
        self.training_stats['losses'] = model.get('losses', 0)
        self.training_max_episodes = model.get('episode', 0) + 5000  # Mais 5000 episódios
        self.training_stats['total_episodes'] = self.training_max_episodes
        
        # Configurar mapa (manter o mesmo do modelo ou usar arena pequena)
        self.selected_map = list(ALL_MAPS.values())[0]
        self.current_map = self.selected_map
        self._setup_map_systems(self.selected_map)
        
        # Configurar câmera para mostrar arena toda
        self.cam_target_mode = "center"
        self.cam_zoom = 1.0
        self._update_camera()
        
        # Limpar históricos
        self.reward_history.clear()
        self.win_rate_history.clear()
        
        # Limpar buffer de experiência
        self._clear_experience_buffer()
        
        # Construir pool de oponentes para rodízio
        if self.training_multi_opponent:
            self._build_opponent_pool()
        
        # Iniciar
        self.menu_state = "training"
        self.training_active = True
        self._reset_training_episode()
        
        self.message = f"Continuando treinamento: {model['agent_class']} + {model['agent_weapon']}"
        self.message_timer = 2.0
    
    def _delete_model(self, model_idx: int):
        """Apaga um modelo"""
        if model_idx >= len(self.available_models):
            return
        
        model = self.available_models[model_idx]
        
        # Apagar arquivo JSON
        try:
            if os.path.exists(model['path']):
                os.remove(model['path'])
        except Exception as e:
            print(f"Erro ao apagar JSON: {e}")
        
        # Apagar arquivo de pesos se existir
        if model.get('weights_file'):
            weights_path = os.path.join("models", model['weights_file'])
            try:
                if os.path.exists(weights_path):
                    os.remove(weights_path)
            except Exception as e:
                print(f"Erro ao apagar pesos: {e}")
        
        # Atualizar lista
        self.available_models = self._scan_models()
        self.message = f"✅ Modelo apagado: {model['filename']}"
        self.message_timer = 2.0

    # ==========================================================================
    # UPDATE
    # ==========================================================================
    
    def update(self, dt: float):
        """Atualiza o estado do jogo"""
        # Atualizar mensagem de feedback
        if self.message_timer > 0:
            self.message_timer -= dt
            if self.message_timer <= 0:
                self.message = ""
        
        # Estados de menu não precisam update de jogo
        if self.menu_state in ["main_menu", "size_select", "map_select", "agent_config", "training_menu"]:
            return
        
        # Modo de treinamento
        if self.menu_state == "training":
            self._update_training(dt)
            return
        
        if self.paused or self.game_over:
            return
        
        # Slow motion
        if self.slow_motion_timer > 0:
            self.slow_motion_timer -= dt
            if self.slow_motion_timer <= 0:
                self.slow_motion = False
        
        effective_dt = dt * self.slow_motion_scale if self.slow_motion else dt
        
        # Guardar vida anterior para detectar dano
        health_before = {id(e): e.health for e in self.entities if e.is_alive()}
        
        # Atualizar entidades
        for entity in self.entities:
            if entity.is_alive():
                entity.update(effective_dt)
        
        # Verificar armadilhas do Trapper
        for entity in self.entities:
            if entity.is_alive() and hasattr(entity, 'check_traps') and hasattr(entity, 'traps'):
                enemies = [e for e in self.entities if e.is_alive() and e.team != entity.team]
                entity.check_traps(enemies)
        
        # Física
        self.physics.handle_collisions(self.entities)
        
        # Colisão com obstáculos e limites
        if self.current_map:
            arena_rect = pygame.Rect(0, 0, self.current_map.width, self.current_map.height)
            
            for entity in self.entities:
                if entity.is_alive():
                    if self.obstacle_manager:
                        self.obstacle_manager.resolve_collision(entity, effective_dt)
                    self.physics.constrain_to_arena(entity, arena_rect)
            
            # Colisão de projéteis
            self.physics.check_projectiles_arena_collision(self.entities, arena_rect)
        
        # Detectar dano para slow motion
        for entity in self.entities:
            entity_id = id(entity)
            if entity_id in health_before:
                if entity.health < health_before[entity_id]:
                    self.slow_motion = True
                    self.slow_motion_timer = self.slow_motion_duration
                    break
        
        # Verificar fim de jogo
        blue_alive = [e for e in self.blue_team if e.is_alive()]
        red_alive = [e for e in self.red_team if e.is_alive()]
        
        if not blue_alive or not red_alive:
            self.game_over = True
            if blue_alive:
                self.winner_team = "blue"
            elif red_alive:
                self.winner_team = "red"
            else:
                self.winner_team = None  # Empate
    
    # ==========================================================================
    # DRAW
    # ==========================================================================
    
    def draw(self):
        """Desenha o jogo"""
        screen.fill(DARK_GRAY)
        
        if self.menu_state == "main_menu":
            self._draw_main_menu()
        elif self.menu_state == "size_select":
            self._draw_size_select()
        elif self.menu_state == "map_select":
            self._draw_map_select()
        elif self.menu_state == "agent_config":
            self._draw_agent_config()
            # Overlay de seleção de modelo
            if self.model_select_active:
                self._draw_model_select()
        elif self.menu_state == "training_menu":
            self._draw_training_menu()
        elif self.menu_state == "training":
            self._draw_training()
        elif self.menu_state == "playing":
            self._draw_game()
        
        # Mensagem de feedback
        if self.message:
            msg_surf = fonts['medium'].render(self.message, True, YELLOW)
            msg_rect = msg_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
            screen.blit(msg_surf, msg_rect)
        
        pygame.display.flip()
    
    def _draw_main_menu(self):
        """Desenha o menu principal"""
        # Título
        title = fonts['title'].render("CIRCLE WARRIORS", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(title, title_rect)
        
        subtitle = fonts['medium'].render("Weapon Ball Fight", True, GRAY)
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 160))
        screen.blit(subtitle, sub_rect)
        
        # Opções
        options = [
            ("🎮  PARTIDA", "Jogue uma batalha"),
            ("🧠  TREINO", "Treine modelos de IA"),
        ]
        
        start_y = 280
        for i, (text, desc) in enumerate(options):
            y = start_y + i * 100
            
            # Highlight se selecionado
            if i == self.menu_cursor:
                pygame.draw.rect(screen, (50, 50, 70), 
                               (SCREEN_WIDTH // 2 - 200, y - 10, 400, 70), border_radius=10)
                pygame.draw.rect(screen, BLUE,
                               (SCREEN_WIDTH // 2 - 200, y - 10, 400, 70), 3, border_radius=10)
                color = WHITE
            else:
                color = GRAY
            
            text_surf = fonts['large'].render(text, True, color)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, y + 10))
            screen.blit(text_surf, text_rect)
            
            desc_surf = fonts['small'].render(desc, True, GRAY)
            desc_rect = desc_surf.get_rect(center=(SCREEN_WIDTH // 2, y + 40))
            screen.blit(desc_surf, desc_rect)
        
        # Instruções
        inst = fonts['small'].render("↑↓ Navegar  |  ENTER Selecionar  |  ESC Sair", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        screen.blit(inst, inst_rect)
    
    def _draw_size_select(self):
        """Desenha seleção de tamanho"""
        # Título
        mode_text = "PARTIDA" if self.selected_mode == "match" else "TREINAMENTO"
        title = fonts['large'].render(f"Selecione o Tamanho - {mode_text}", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
        screen.blit(title, title_rect)
        
        # Opções de tamanho
        sizes = [
            ("1 vs 1", "Duelo individual"),
            ("2 vs 2", "Batalha em dupla"),
            ("3 vs 3", "Combate tático"),
            ("5 vs 5", "Guerra em equipe"),
        ]
        
        start_y = 180
        for i, (text, desc) in enumerate(sizes):
            y = start_y + i * 90
            
            if i == self.menu_cursor:
                pygame.draw.rect(screen, (50, 50, 70),
                               (SCREEN_WIDTH // 2 - 180, y - 5, 360, 70), border_radius=10)
                pygame.draw.rect(screen, GREEN,
                               (SCREEN_WIDTH // 2 - 180, y - 5, 360, 70), 3, border_radius=10)
                color = WHITE
            else:
                color = GRAY
            
            text_surf = fonts['large'].render(text, True, color)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, y + 15))
            screen.blit(text_surf, text_rect)
            
            desc_surf = fonts['small'].render(desc, True, GRAY)
            desc_rect = desc_surf.get_rect(center=(SCREEN_WIDTH // 2, y + 45))
            screen.blit(desc_surf, desc_rect)
        
        # Instruções
        inst = fonts['small'].render("↑↓ Navegar  |  ENTER Selecionar  |  ESC Voltar", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        screen.blit(inst, inst_rect)
    
    def _draw_map_select(self):
        """Desenha seleção de mapa"""
        # Título
        title = fonts['large'].render("Selecione o Mapa", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 50))
        screen.blit(title, title_rect)
        
        all_maps = list(ALL_MAPS.values())
        current_map = all_maps[self.selected_map_idx]
        
        # Preview do mapa (box colorido)
        preview_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, 100, 400, 250)
        pygame.draw.rect(screen, current_map.bg_color, preview_rect)
        pygame.draw.rect(screen, current_map.floor_color, 
                        preview_rect.inflate(-40, -40))
        pygame.draw.rect(screen, current_map.border_color, preview_rect, 4)
        
        # Indicador de tamanho
        size_text = "GRANDE" if current_map.is_large else "PEQUENO"
        size_color = PURPLE if current_map.is_large else GREEN
        size_surf = fonts['medium'].render(size_text, True, size_color)
        screen.blit(size_surf, (preview_rect.x + 10, preview_rect.y + 10))
        
        # Nome do mapa
        name_surf = fonts['large'].render(current_map.name, True, WHITE)
        name_rect = name_surf.get_rect(center=(SCREEN_WIDTH // 2, 380))
        screen.blit(name_surf, name_rect)
        
        # Descrição
        desc_surf = fonts['small'].render(current_map.description, True, GRAY)
        desc_rect = desc_surf.get_rect(center=(SCREEN_WIDTH // 2, 420))
        screen.blit(desc_surf, desc_rect)
        
        # Dimensões
        dim_text = f"Tamanho: {current_map.width} x {current_map.height} pixels"
        dim_surf = fonts['small'].render(dim_text, True, GRAY)
        dim_rect = dim_surf.get_rect(center=(SCREEN_WIDTH // 2, 450))
        screen.blit(dim_surf, dim_rect)
        
        # Info de fog/clima
        info_parts = ["🌫️ Fog of War ATIVO"]
        if current_map.has_weather:
            weather_names = {"snow": "❄️ Neve", "rain": "🌧️ Chuva", 
                           "sandstorm": "🌪️ Tempestade", "fog": "🌫️ Neblina"}
            info_parts.append(weather_names.get(current_map.weather_type, ""))
        
        info_text = "  |  ".join(info_parts)
        info_surf = fonts['small'].render(info_text, True, (150, 200, 255))
        info_rect = info_surf.get_rect(center=(SCREEN_WIDTH // 2, 480))
        screen.blit(info_surf, info_rect)
        
        # Setas de navegação
        arrow_y = preview_rect.centery
        pygame.draw.polygon(screen, WHITE, [
            (preview_rect.x - 50, arrow_y),
            (preview_rect.x - 30, arrow_y - 20),
            (preview_rect.x - 30, arrow_y + 20)
        ])
        pygame.draw.polygon(screen, WHITE, [
            (preview_rect.right + 50, arrow_y),
            (preview_rect.right + 30, arrow_y - 20),
            (preview_rect.right + 30, arrow_y + 20)
        ])
        
        # Indicador de posição
        indicator_y = 520
        for i in range(len(all_maps)):
            color = WHITE if i == self.selected_map_idx else GRAY
            radius = 6 if i == self.selected_map_idx else 4
            x = SCREEN_WIDTH // 2 - (len(all_maps) * 15) // 2 + i * 15
            pygame.draw.circle(screen, color, (x, indicator_y), radius)
        
        # Contador
        counter = fonts['small'].render(f"{self.selected_map_idx + 1} / {len(all_maps)}", True, GRAY)
        counter_rect = counter.get_rect(center=(SCREEN_WIDTH // 2, 550))
        screen.blit(counter, counter_rect)
        
        # Instruções
        inst = fonts['small'].render("←→ Navegar  |  ENTER Selecionar  |  ESC Voltar", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        screen.blit(inst, inst_rect)
    
    def _draw_agent_config(self):
        """Desenha configuração de agentes"""
        # Título
        title = fonts['large'].render("Configure os Agentes", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 40))
        screen.blit(title, title_rect)
        
        # Subtítulo com mapa selecionado
        if self.selected_map:
            map_info = fonts['small'].render(
                f"Mapa: {self.selected_map.name} | {self.selected_size}v{self.selected_size}",
                True, GRAY
            )
            map_rect = map_info.get_rect(center=(SCREEN_WIDTH // 2, 70))
            screen.blit(map_info, map_rect)
        
        # Dividir tela em dois times
        blue_x = SCREEN_WIDTH // 4
        red_x = 3 * SCREEN_WIDTH // 4
        
        # Cabeçalhos dos times
        blue_title = fonts['medium'].render("TIME AZUL", True, BLUE)
        blue_rect = blue_title.get_rect(center=(blue_x, 110))
        screen.blit(blue_title, blue_rect)
        
        red_title = fonts['medium'].render("TIME VERMELHO", True, RED)
        red_rect = red_title.get_rect(center=(red_x, 110))
        screen.blit(red_title, red_rect)
        
        # Desenhar agentes de cada time
        self._draw_team_agents(self.blue_agents, blue_x, 150, "blue")
        self._draw_team_agents(self.red_agents, red_x, 150, "red")
        
        # Campo atual sendo editado
        field_names = ["CLASSE", "ARMA", "CONTROLE"]
        field_text = f"Editando: {field_names[self.agent_config_field]} (TAB para alternar)"
        field_surf = fonts['small'].render(field_text, True, YELLOW)
        field_rect = field_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100))
        screen.blit(field_surf, field_rect)
        
        # Instruções
        inst = fonts['small'].render(
            "↑↓ Agente  |  ←→ Mudar valor  |  TAB Campo  |  ENTER Iniciar  |  ESC Voltar",
            True, GRAY
        )
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
        screen.blit(inst, inst_rect)
    
    def _draw_team_agents(self, agents: List[AgentConfig], center_x: int, start_y: int, team: str):
        """Desenha lista de agentes de um time"""
        for i, agent in enumerate(agents):
            y = start_y + i * 100
            
            # Verificar se este agente está selecionado
            is_selected = (self.configuring_team == team and self.agent_config_cursor == i)
            
            # Box do agente
            box_rect = pygame.Rect(center_x - 180, y, 360, 85)
            
            if is_selected:
                pygame.draw.rect(screen, (40, 50, 60), box_rect, border_radius=8)
                border_color = BLUE if team == "blue" else RED
                pygame.draw.rect(screen, border_color, box_rect, 3, border_radius=8)
            else:
                pygame.draw.rect(screen, (30, 35, 40), box_rect, border_radius=8)
                pygame.draw.rect(screen, (60, 60, 70), box_rect, 1, border_radius=8)
            
            # Número do agente
            num_surf = fonts['medium'].render(f"#{i + 1}", True, GRAY)
            screen.blit(num_surf, (box_rect.x + 10, box_rect.y + 5))
            
            # Classe
            cls = ClassRegistry.get_all().get(agent.class_id)
            cls_name = cls.display_name if cls else agent.class_id
            cls_color = WHITE if (is_selected and self.agent_config_field == 0) else GRAY
            cls_surf = fonts['small'].render(f"Classe: {cls_name}", True, cls_color)
            screen.blit(cls_surf, (box_rect.x + 50, box_rect.y + 10))
            
            # Arma
            weapon = WeaponRegistry.get_all().get(agent.weapon_id)
            weapon_name = weapon.display_name if weapon else agent.weapon_id
            weapon_color = WHITE if (is_selected and self.agent_config_field == 1) else GRAY
            weapon_surf = fonts['small'].render(f"Arma: {weapon_name}", True, weapon_color)
            screen.blit(weapon_surf, (box_rect.x + 50, box_rect.y + 35))
            
            # Controlador
            ctrl_names = {
                "ai": "🤖 IA Estratégica",
                "player1": "🎮 Jogador 1 (WASD)",
                "player2": "🎮 Jogador 2 (Setas)",
                "model": "🧠 Modelo Treinado"
            }
            ctrl_text = ctrl_names.get(agent.controller_type, agent.controller_type)
            ctrl_color = WHITE if (is_selected and self.agent_config_field == 2) else GRAY
            ctrl_surf = fonts['small'].render(f"Controle: {ctrl_text}", True, ctrl_color)
            screen.blit(ctrl_surf, (box_rect.x + 50, box_rect.y + 60))
    
    def _draw_training_menu(self):
        """Desenha o menu de gerenciamento de treinamento"""
        if self.training_menu_state == "main":
            self._draw_training_menu_main()
        elif self.training_menu_state == "new_model":
            self._draw_training_menu_new_model()
        elif self.training_menu_state == "view_model":
            self._draw_training_menu_view_model()
        elif self.training_menu_state == "confirm_delete":
            self._draw_training_menu_confirm_delete()
    
    def _draw_training_menu_main(self):
        """Desenha menu principal de treinamento"""
        # Título
        title = fonts['title'].render("TREINAMENTO DE IA", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 60))
        screen.blit(title, title_rect)
        
        subtitle = fonts['small'].render("Gerencie e treine modelos de Inteligência Artificial", True, GRAY)
        sub_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(subtitle, sub_rect)
        
        # Opções fixas
        y = 150
        options = [
            ("➕  NOVO MODELO", "Criar e treinar um novo modelo de IA"),
            ("⬅️  VOLTAR", "Retornar ao menu principal"),
            ("─" * 30, ""),  # Separador
        ]
        
        for i, (text, desc) in enumerate(options):
            is_selected = (self.training_menu_cursor == i)
            
            if i == 2:  # Separador
                sep = fonts['small'].render(text, True, GRAY)
                sep_rect = sep.get_rect(center=(SCREEN_WIDTH // 2, y + 10))
                screen.blit(sep, sep_rect)
                y += 30
                continue
            
            if is_selected:
                pygame.draw.rect(screen, (50, 50, 70),
                               (SCREEN_WIDTH // 2 - 250, y - 5, 500, 50), border_radius=8)
                pygame.draw.rect(screen, GREEN if i == 0 else BLUE,
                               (SCREEN_WIDTH // 2 - 250, y - 5, 500, 50), 2, border_radius=8)
                color = WHITE
            else:
                color = GRAY
            
            text_surf = fonts['medium'].render(text, True, color)
            text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, y + 10))
            screen.blit(text_surf, text_rect)
            
            if desc:
                desc_surf = fonts['small'].render(desc, True, (100, 100, 100))
                desc_rect = desc_surf.get_rect(center=(SCREEN_WIDTH // 2, y + 32))
                screen.blit(desc_surf, desc_rect)
            
            y += 60
        
        # Lista de modelos
        if self.available_models:
            models_title = fonts['medium'].render(f"📁 Modelos Salvos ({len(self.available_models)})", True, WHITE)
            screen.blit(models_title, (SCREEN_WIDTH // 2 - 250, y))
            y += 35
            
            # Mostrar até 6 modelos
            visible_start = max(0, (self.training_menu_cursor - 3) - 2) if self.training_menu_cursor >= 3 else 0
            visible_end = min(len(self.available_models), visible_start + 6)
            
            for i in range(visible_start, visible_end):
                model = self.available_models[i]
                menu_idx = i + 3  # Offset para opções fixas
                is_selected = (self.training_menu_cursor == menu_idx)
                
                box_rect = pygame.Rect(SCREEN_WIDTH // 2 - 250, y, 500, 55)
                
                if is_selected:
                    pygame.draw.rect(screen, (40, 50, 60), box_rect, border_radius=5)
                    pygame.draw.rect(screen, PURPLE, box_rect, 2, border_radius=5)
                else:
                    pygame.draw.rect(screen, (30, 35, 40), box_rect, border_radius=5)
                
                # Info do modelo
                cls_name = model.get('agent_class', 'unknown')
                wpn_name = model.get('agent_weapon', 'unknown')
                episode = model.get('episode', 0)
                win_rate = model.get('win_rate', 0)
                
                name_text = f"🤖 {cls_name} + {wpn_name}"
                stats_text = f"Ep: {episode} | Win: {win_rate:.1f}%"
                
                color = WHITE if is_selected else GRAY
                name_surf = fonts['small'].render(name_text, True, color)
                screen.blit(name_surf, (box_rect.x + 10, box_rect.y + 8))
                
                stats_surf = fonts['small'].render(stats_text, True, (100, 150, 100) if is_selected else (80, 80, 80))
                screen.blit(stats_surf, (box_rect.x + 10, box_rect.y + 30))
                
                # Indicador de rede neural
                if model.get('has_neural_weights'):
                    nn_surf = fonts['small'].render("🧠", True, GREEN)
                    screen.blit(nn_surf, (box_rect.right - 40, box_rect.y + 18))
                
                y += 60
        else:
            no_models = fonts['medium'].render("Nenhum modelo salvo ainda", True, GRAY)
            no_rect = no_models.get_rect(center=(SCREEN_WIDTH // 2, y + 50))
            screen.blit(no_models, no_rect)
        
        # Instruções
        inst = fonts['small'].render("↑↓ Navegar | ENTER Selecionar | DEL Apagar modelo | ESC Voltar", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        screen.blit(inst, inst_rect)
    
    def _draw_training_menu_new_model(self):
        """Desenha tela de criação de novo modelo"""
        # Título
        title = fonts['large'].render("CRIAR NOVO MODELO", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 80))
        screen.blit(title, title_rect)
        
        # Pegar nome do mapa atual
        map_key = self.training_map_list[self.new_model_map]
        map_name = ALL_MAPS[map_key].name if map_key in ALL_MAPS else map_key
        
        # Campos de configuração
        fields = [
            ("Classe", self.available_classes[self.new_model_class].upper(), 0),
            ("Arma", self.available_weapons[self.new_model_weapon].upper(), 1),
            ("Mapa", map_name, 2),
            ("Episódios", str(self.new_model_episodes), 3),
            ("▶️  INICIAR TREINAMENTO", "", 4),
        ]
        
        y = 150
        for text, value, field_idx in fields:
            is_selected = (self.new_model_field == field_idx)
            
            box_rect = pygame.Rect(SCREEN_WIDTH // 2 - 200, y, 400, 45)
            
            if is_selected:
                pygame.draw.rect(screen, (50, 60, 70), box_rect, border_radius=8)
                pygame.draw.rect(screen, GREEN if field_idx == 4 else YELLOW, box_rect, 2, border_radius=8)
                color = WHITE
            else:
                pygame.draw.rect(screen, (35, 40, 45), box_rect, border_radius=8)
                color = GRAY
            
            if field_idx < 4:
                # Campo com valor
                label = fonts['small'].render(text + ":", True, color)
                screen.blit(label, (box_rect.x + 15, box_rect.y + 8))
                
                value_surf = fonts['small'].render(value, True, color)
                value_rect = value_surf.get_rect(center=(box_rect.centerx + 60, box_rect.centery))
                screen.blit(value_surf, value_rect)
                
                # Setas
                if is_selected:
                    arrow_l = fonts['medium'].render("◀", True, YELLOW)
                    arrow_r = fonts['medium'].render("▶", True, YELLOW)
                    screen.blit(arrow_l, (box_rect.x + 100, box_rect.y + 10))
                    screen.blit(arrow_r, (box_rect.right - 40, box_rect.y + 10))
            else:
                # Botão
                btn_text = fonts['medium'].render(text, True, color)
                btn_rect = btn_text.get_rect(center=box_rect.center)
                screen.blit(btn_text, btn_rect)
            
            y += 55
        
        # Preview
        preview_y = y + 15
        preview_text = f"Modelo: {self.available_classes[self.new_model_class]} + {self.available_weapons[self.new_model_weapon]}"
        preview_surf = fonts['small'].render(preview_text, True, (100, 150, 200))
        preview_rect = preview_surf.get_rect(center=(SCREEN_WIDTH // 2, preview_y))
        screen.blit(preview_surf, preview_rect)
        
        info_text = f"Treinará contra {len(self.available_classes) * len(self.available_weapons)} combinações de oponentes"
        info_surf = fonts['small'].render(info_text, True, GRAY)
        info_rect = info_surf.get_rect(center=(SCREEN_WIDTH // 2, preview_y + 25))
        screen.blit(info_surf, info_rect)
        
        # Instruções
        inst = fonts['small'].render("↑↓ Campo | ←→ Valor | ENTER Confirmar | ESC Voltar", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        screen.blit(inst, inst_rect)
    
    def _draw_training_menu_view_model(self):
        """Desenha visualização detalhada de um modelo"""
        if not self.available_models or self.viewing_model_idx >= len(self.available_models):
            return
        
        model = self.available_models[self.viewing_model_idx]
        
        # Título
        title = fonts['large'].render("DETALHES DO MODELO", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 60))
        screen.blit(title, title_rect)
        
        # Navegação
        nav_text = f"◀  {self.viewing_model_idx + 1} / {len(self.available_models)}  ▶"
        nav_surf = fonts['medium'].render(nav_text, True, GRAY)
        nav_rect = nav_surf.get_rect(center=(SCREEN_WIDTH // 2, 100))
        screen.blit(nav_surf, nav_rect)
        
        # Painel de info
        panel_x = SCREEN_WIDTH // 2 - 300
        panel_y = 140
        panel_w = 600
        panel_h = 400
        
        pygame.draw.rect(screen, (35, 40, 50), (panel_x, panel_y, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(screen, PURPLE, (panel_x, panel_y, panel_w, panel_h), 2, border_radius=10)
        
        # Nome do modelo
        model_name = f"{model.get('agent_class', '?')} + {model.get('agent_weapon', '?')}"
        name_surf = fonts['large'].render(model_name, True, WHITE)
        screen.blit(name_surf, (panel_x + 20, panel_y + 15))
        
        # Arquivo
        file_surf = fonts['small'].render(model.get('filename', 'N/A'), True, GRAY)
        screen.blit(file_surf, (panel_x + 20, panel_y + 55))
        
        # Estatísticas
        y = panel_y + 100
        stats = [
            ("Episódios", f"{model.get('episode', 0)} / {model.get('total_episodes', 0)}"),
            ("Vitórias", str(model.get('wins', 0))),
            ("Derrotas", str(model.get('losses', 0))),
            ("Taxa de Vitória", f"{model.get('win_rate', 0):.1f}%"),
            ("Reward Médio", f"{model.get('avg_reward', 0):.2f}"),
            ("Melhor Reward", f"{model.get('best_reward', 0):.2f}"),
            ("Tempo de Treino", f"{model.get('training_time', 0):.1f}s"),
            ("Multi-oponente", "Sim" if model.get('multi_opponent') else "Não"),
            ("Rede Neural", "✅ Sim" if model.get('has_neural_weights') else "❌ Não"),
        ]
        
        col1_x = panel_x + 30
        col2_x = panel_x + panel_w // 2 + 20
        
        for i, (label, value) in enumerate(stats):
            x = col1_x if i < 5 else col2_x
            row_y = y + (i % 5) * 35
            
            label_surf = fonts['small'].render(label + ":", True, GRAY)
            screen.blit(label_surf, (x, row_y))
            
            value_color = GREEN if "✅" in value else (RED if "❌" in value else WHITE)
            value_surf = fonts['small'].render(value, True, value_color)
            screen.blit(value_surf, (x + 150, row_y))
        
        # Gráfico simples de progresso (barra)
        bar_y = panel_y + panel_h - 60
        bar_w = panel_w - 60
        bar_h = 20
        
        pygame.draw.rect(screen, (50, 50, 60), (panel_x + 30, bar_y, bar_w, bar_h), border_radius=5)
        
        progress = model.get('episode', 0) / max(1, model.get('total_episodes', 1))
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            pygame.draw.rect(screen, GREEN, (panel_x + 30, bar_y, fill_w, bar_h), border_radius=5)
        
        progress_text = f"Progresso: {progress * 100:.1f}%"
        progress_surf = fonts['small'].render(progress_text, True, WHITE)
        progress_rect = progress_surf.get_rect(center=(panel_x + panel_w // 2, bar_y + bar_h + 15))
        screen.blit(progress_surf, progress_rect)
        
        # Instruções
        inst = fonts['small'].render("←→ Navegar | C Continuar treino | DEL Apagar | ESC Voltar", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))
        screen.blit(inst, inst_rect)
    
    def _draw_training_menu_confirm_delete(self):
        """Desenha confirmação de exclusão"""
        if not self.available_models or self.viewing_model_idx >= len(self.available_models):
            return
        
        model = self.available_models[self.viewing_model_idx]
        
        # Overlay escuro
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))
        
        # Painel
        panel_w = 450
        panel_h = 200
        panel_x = (SCREEN_WIDTH - panel_w) // 2
        panel_y = (SCREEN_HEIGHT - panel_h) // 2
        
        pygame.draw.rect(screen, (40, 30, 30), (panel_x, panel_y, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(screen, RED, (panel_x, panel_y, panel_w, panel_h), 3, border_radius=10)
        
        # Título
        title = fonts['large'].render("⚠️ CONFIRMAR EXCLUSÃO", True, RED)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, panel_y + 35))
        screen.blit(title, title_rect)
        
        # Mensagem
        msg = fonts['medium'].render(f"Apagar modelo: {model.get('agent_class', '?')} + {model.get('agent_weapon', '?')}?", True, WHITE)
        msg_rect = msg.get_rect(center=(SCREEN_WIDTH // 2, panel_y + 80))
        screen.blit(msg, msg_rect)
        
        warn = fonts['small'].render("Esta ação não pode ser desfeita!", True, YELLOW)
        warn_rect = warn.get_rect(center=(SCREEN_WIDTH // 2, panel_y + 110))
        screen.blit(warn, warn_rect)
        
        # Botões
        btn_y = panel_y + 150
        
        # Não
        no_rect = pygame.Rect(panel_x + 50, btn_y, 150, 35)
        no_selected = not self.delete_confirm
        pygame.draw.rect(screen, (60, 80, 60) if no_selected else (40, 50, 40), no_rect, border_radius=5)
        if no_selected:
            pygame.draw.rect(screen, GREEN, no_rect, 2, border_radius=5)
        no_text = fonts['medium'].render("CANCELAR", True, WHITE if no_selected else GRAY)
        no_text_rect = no_text.get_rect(center=no_rect.center)
        screen.blit(no_text, no_text_rect)
        
        # Sim
        yes_rect = pygame.Rect(panel_x + panel_w - 200, btn_y, 150, 35)
        yes_selected = self.delete_confirm
        pygame.draw.rect(screen, (100, 50, 50) if yes_selected else (60, 40, 40), yes_rect, border_radius=5)
        if yes_selected:
            pygame.draw.rect(screen, RED, yes_rect, 2, border_radius=5)
        yes_text = fonts['medium'].render("APAGAR", True, WHITE if yes_selected else GRAY)
        yes_text_rect = yes_text.get_rect(center=yes_rect.center)
        screen.blit(yes_text, yes_text_rect)
        
        # Instruções
        inst = fonts['small'].render("←→ Selecionar | ENTER Confirmar | ESC Cancelar", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, panel_y + panel_h + 30))
        screen.blit(inst, inst_rect)
    
    def _draw_model_select(self):
        """Desenha overlay de seleção de modelo"""
        # Overlay escuro
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 200))
        screen.blit(overlay, (0, 0))
        
        # Painel central
        panel_w = 500
        panel_h = 400
        panel_x = (SCREEN_WIDTH - panel_w) // 2
        panel_y = (SCREEN_HEIGHT - panel_h) // 2
        
        pygame.draw.rect(screen, (30, 35, 45), (panel_x, panel_y, panel_w, panel_h), border_radius=10)
        pygame.draw.rect(screen, PURPLE, (panel_x, panel_y, panel_w, panel_h), 3, border_radius=10)
        
        # Título
        title = fonts['large'].render("Selecione o Modelo", True, WHITE)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, panel_y + 30))
        screen.blit(title, title_rect)
        
        # Lista de modelos
        if not self.available_models:
            no_models = fonts['medium'].render("Nenhum modelo encontrado", True, GRAY)
            no_rect = no_models.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            screen.blit(no_models, no_rect)
        else:
            y = panel_y + 70
            visible_models = min(6, len(self.available_models))
            start_idx = max(0, self.model_select_cursor - 2)
            end_idx = min(len(self.available_models), start_idx + visible_models)
            
            for i in range(start_idx, end_idx):
                model = self.available_models[i]
                is_selected = (i == self.model_select_cursor)
                
                # Box
                box_y = y + (i - start_idx) * 50
                box_rect = pygame.Rect(panel_x + 20, box_y, panel_w - 40, 45)
                
                if is_selected:
                    pygame.draw.rect(screen, (50, 60, 80), box_rect, border_radius=5)
                    pygame.draw.rect(screen, PURPLE, box_rect, 2, border_radius=5)
                else:
                    pygame.draw.rect(screen, (40, 45, 55), box_rect, border_radius=5)
                
                # Info do modelo
                cls_name = model.get('agent_class', 'unknown')
                wpn_name = model.get('agent_weapon', 'unknown')
                episode = model.get('episode', 0)
                
                text = f"{cls_name} + {wpn_name} (Ep {episode})"
                color = WHITE if is_selected else GRAY
                text_surf = fonts['small'].render(text, True, color)
                screen.blit(text_surf, (box_rect.x + 10, box_rect.y + 5))
                
                # Nome do arquivo
                filename = model.get('filename', '')[:40]
                file_surf = fonts['small'].render(filename, True, (100, 100, 120))
                screen.blit(file_surf, (box_rect.x + 10, box_rect.y + 25))
        
        # Instruções
        inst = fonts['small'].render("↑↓ Navegar | ENTER Selecionar | ESC Cancelar", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, panel_y + panel_h - 25))
        screen.blit(inst, inst_rect)
    
    def _draw_game(self):
        """Desenha o jogo em andamento"""
        if not self.current_map:
            return
        
        # Atualizar câmera
        self._update_camera()
        
        # Desenhar mundo
        self._draw_game_world()
        
        # UI sempre por cima
        self._draw_game_ui()
        
        # Slow motion indicator
        if self.slow_motion:
            pulse = int(abs(math.sin(pygame.time.get_ticks() / 100)) * 100) + 100
            pygame.draw.rect(screen, (pulse, pulse // 2, 0), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 6)
        
        # Game over overlay
        if self.game_over:
            self._draw_game_over()
        elif self.paused:
            self._draw_paused()
    
    def _draw_game_world(self):
        """Desenha o mundo do jogo com a câmera simples (Otimizado)"""
        if not self.current_map:
            return
            
        map_cfg = self.current_map
        
        # Calcular viewport
        view_w = SCREEN_WIDTH / self.cam_zoom
        view_h = SCREEN_HEIGHT / self.cam_zoom
        
        # Criar surface do viewport
        view_surf = pygame.Surface((int(view_w), int(view_h)))
        view_surf.fill(map_cfg.bg_color)
        
        cam_offset = (self.cam_x, self.cam_y)
        
        # Grid (opcional)
        if map_cfg.is_large:
            grid_size = 200
            start_x = int(self.cam_x // grid_size) * grid_size
            end_x = int((self.cam_x + view_w) // grid_size + 1) * grid_size
            start_y = int(self.cam_y // grid_size) * grid_size
            end_y = int((self.cam_y + view_h) // grid_size + 1) * grid_size
            
            for x in range(start_x, end_x + 1, grid_size):
                draw_x = x - self.cam_x
                pygame.draw.line(view_surf, map_cfg.accent_color, (draw_x, 0), (draw_x, view_h), 1)
            for y in range(start_y, end_y + 1, grid_size):
                draw_y = y - self.cam_y
                pygame.draw.line(view_surf, map_cfg.accent_color, (0, draw_y), (view_w, draw_y), 1)
        else:
            # Arena pequena
            play_rect = pygame.Rect(50 - self.cam_x, 50 - self.cam_y, map_cfg.width - 100, map_cfg.height - 100)
            pygame.draw.rect(view_surf, map_cfg.floor_color, play_rect)
            pygame.draw.rect(view_surf, map_cfg.border_color, play_rect, 3)
            
        # Obstáculos
        if self.obstacle_manager:
            self.obstacle_manager.draw(view_surf, cam_offset)
            
        # Entidades
        for entity in self.entities:
            # Culling simples
            if (entity.x + entity.radius < self.cam_x or 
                entity.x - entity.radius > self.cam_x + view_w or
                entity.y + entity.radius < self.cam_y or
                entity.y - entity.radius > self.cam_y + view_h):
                continue
                
            # Shift temporário
            original_x, original_y = entity.x, entity.y
            entity.x -= self.cam_x
            entity.y -= self.cam_y
            
            # Draw
            if entity.__class__.__name__ == 'Trapper':
                entity.draw(view_surf, cam_offset=cam_offset)
            else:
                entity.draw(view_surf)
                
            # Restore
            entity.x, entity.y = original_x, original_y
            
        # Fog of War
        vision_target = None
        if self.menu_state == "training":
            vision_target = self.training_entity
        elif self.cam_target_mode == "p1":
            for entity in self.blue_team:
                if hasattr(entity, 'controller') and isinstance(entity.controller, PlayerController):
                    vision_target = entity
                    break
            if not vision_target and self.blue_team:
                vision_target = self.blue_team[0]
        elif self.cam_target_mode == "p2":
             if self.red_team:
                vision_target = self.red_team[0]
        
        if self.fog_of_war and vision_target:
            self.fog_of_war.draw_fog(view_surf, vision_target.team, self.entities, cam_offset)
            
        # Scale to screen
        if abs(self.cam_zoom - 1.0) < 0.01:
            screen.blit(view_surf, (0, 0))
        else:
            scaled = pygame.transform.scale(view_surf, (SCREEN_WIDTH, SCREEN_HEIGHT))
            screen.blit(scaled, (0, 0))
            
        # Borda do mapa (visualização)
        screen_x = -self.cam_x * self.cam_zoom
        screen_y = -self.cam_y * self.cam_zoom
        scaled_w = map_cfg.width * self.cam_zoom
        scaled_h = map_cfg.height * self.cam_zoom
        pygame.draw.rect(screen, map_cfg.border_color, 
                        (screen_x, screen_y, scaled_w, scaled_h), 3)
        
        # Minimapa
        self._draw_minimap()
    
    def _draw_minimap(self):
        """Desenha minimapa no canto"""
        if not self.current_map:
            return
        
        mm_w, mm_h = 150, 150
        mm_x = SCREEN_WIDTH - mm_w - 10
        mm_y = SCREEN_HEIGHT - mm_h - 60
        
        mm_surface = pygame.Surface((mm_w, mm_h), pygame.SRCALPHA)
        mm_surface.fill((0, 0, 0, 180))
        
        scale_x = mm_w / self.current_map.width
        scale_y = mm_h / self.current_map.height
        
        # Obstáculos
        if self.obstacle_manager:
            for obs in self.obstacle_manager.obstacles:
                ox = int(obs.rect.x * scale_x)
                oy = int(obs.rect.y * scale_y)
                ow = max(2, int(obs.rect.width * scale_x))
                oh = max(2, int(obs.rect.height * scale_y))
                pygame.draw.rect(mm_surface, (60, 60, 60), (ox, oy, ow, oh))
        
        # Entidades
        for entity in self.entities:
            if entity.is_alive():
                ex = int(entity.x * scale_x)
                ey = int(entity.y * scale_y)
                color = BLUE if entity.team == "blue" else RED
                pygame.draw.circle(mm_surface, color, (ex, ey), 4)
        
        # Viewport
        vx = int(self.cam_x * scale_x)
        vy = int(self.cam_y * scale_y)
        vw = int((SCREEN_WIDTH / self.cam_zoom) * scale_x)
        vh = int((SCREEN_HEIGHT / self.cam_zoom) * scale_y)
        pygame.draw.rect(mm_surface, WHITE, (vx, vy, vw, vh), 1)
        
        pygame.draw.rect(mm_surface, GRAY, (0, 0, mm_w, mm_h), 2)
        screen.blit(mm_surface, (mm_x, mm_y))
    
    def _draw_game_ui(self):
        """Desenha UI do jogo (vida, etc)"""
        # Info dos times
        blue_alive = sum(1 for e in self.blue_team if e.is_alive())
        red_alive = sum(1 for e in self.red_team if e.is_alive())
        
        # Time azul (esquerda)
        blue_text = f"AZUL: {blue_alive}/{len(self.blue_team)}"
        blue_surf = fonts['medium'].render(blue_text, True, BLUE)
        screen.blit(blue_surf, (20, 10))
        
        # Time vermelho (direita)
        red_text = f"VERMELHO: {red_alive}/{len(self.red_team)}"
        red_surf = fonts['medium'].render(red_text, True, RED)
        red_rect = red_surf.get_rect(topright=(SCREEN_WIDTH - 20, 10))
        screen.blit(red_surf, red_rect)
        
        # Barras de vida individuais (para o time do jogador)
        y_offset = 50
        for entity in self.blue_team:
            if entity.is_alive():
                stats = entity.stats_manager.get_stats()
                health_ratio = entity.health / stats.max_health
                
                # Nome
                name = f"{entity.display_name}"
                name_surf = fonts['small'].render(name, True, WHITE)
                screen.blit(name_surf, (20, y_offset))
                
                # Barra
                bar_w, bar_h = 120, 8
                pygame.draw.rect(screen, (40, 40, 40), (20, y_offset + 20, bar_w, bar_h))
                pygame.draw.rect(screen, BLUE, (20, y_offset + 20, bar_w * health_ratio, bar_h))
                pygame.draw.rect(screen, WHITE, (20, y_offset + 20, bar_w, bar_h), 1)
                
                y_offset += 35
    
            # Dica e status de câmera (HUD)
            cam_status = f"Cam: {self.cam_target_mode} | Zoom: {self.cam_zoom:.2f}"
            status_surf = fonts['small'].render(cam_status, True, YELLOW)
            screen.blit(status_surf, (20, SCREEN_HEIGHT - 50))

            cam_hint = "C: Alvo (P1/P2/Centro) | Z/X ou -/+: Zoom | 0: Reset Zoom"
            hint_surf = fonts['small'].render(cam_hint, True, GRAY)
            hint_rect = hint_surf.get_rect(bottomleft=(20, SCREEN_HEIGHT - 20))
            screen.blit(hint_surf, hint_rect)

    def _draw_game_over(self):
        """Desenha overlay de fim de jogo"""
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        screen.blit(overlay, (0, 0))
        
        if self.winner_team:
            winner_text = f"TIME {self.winner_team.upper()} VENCEU!"
            color = BLUE if self.winner_team == "blue" else RED
        else:
            winner_text = "EMPATE!"
            color = WHITE
        
        text_surf = fonts['title'].render(winner_text, True, color)
        text_rect = text_surf.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
        screen.blit(text_surf, text_rect)
        
        restart = fonts['medium'].render("Pressione R para reiniciar | ESC para menu", True, WHITE)
        restart_rect = restart.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 40))
        screen.blit(restart, restart_rect)
    
    def _draw_paused(self):
        """Desenha overlay de pausa"""
        if not self.current_map:
            screen.fill(DARK_GRAY)
            return
        
        self._draw_world_with_camera()
        inst = fonts['medium'].render("Pressione P para continuar", True, GRAY)
        inst_rect = inst.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
        screen.blit(inst, inst_rect)
    
    # ==========================================================================
    # SISTEMA DE TREINAMENTO COMPLETO
    # ==========================================================================
    
    def _start_training(self):
        """Inicia o treinamento de IA com rede neural"""
        def _draw_world_with_camera(self):
            """Desenha o mundo aplicando zoom e deslocamento da câmera"""
            map_cfg = self.current_map
            target = self.camera_target if self.camera_target else (self.blue_team[0] if self.blue_team else None)
            zoom, origin_x, origin_y = self._get_camera_view(target)
            map_w, map_h = map_cfg.width, map_cfg.height
            view_w, view_h = SCREEN_WIDTH / zoom, SCREEN_HEIGHT / zoom
            self.camera_viewport = (origin_x, origin_y, view_w, view_h, zoom)
        
            world = pygame.Surface((map_w, map_h))
            world.fill(map_cfg.bg_color)
        
            # Grid opcional para mapas grandes
            if map_cfg.is_large:
                grid_size = 200
                for x in range(0, map_w, grid_size):
                    pygame.draw.line(world, map_cfg.accent_color, (x, 0), (x, map_h), 1)
                for y in range(0, map_h, grid_size):
                    pygame.draw.line(world, map_cfg.accent_color, (0, y), (map_w, y), 1)
            else:
                # Área jogável destacada
                play_rect = pygame.Rect(50, 50, map_w - 100, map_h - 100)
                pygame.draw.rect(world, map_cfg.floor_color, play_rect)
                pygame.draw.rect(world, map_cfg.border_color, play_rect, 3)
        
            # Obstáculos
            if self.obstacle_manager:
                self.obstacle_manager.draw(world, (0, 0))
        
            # Entidades
            for entity in self.entities:
                entity.draw(world)
        
            # Fog of war
            if self.fog_of_war and target:
                self.fog_of_war.draw_fog(world, target.team, self.entities, (0, 0))
        
            # Escalar e posicionar
            scaled_world = pygame.transform.scale(world, (int(map_w * zoom), int(map_h * zoom)))
            screen.blit(scaled_world, (-origin_x * zoom, -origin_y * zoom))
        
            # Borda do mapa
            pygame.draw.rect(screen, map_cfg.border_color, 
                             (-origin_x * zoom, -origin_y * zoom, map_w * zoom, map_h * zoom), 3)
        
            # Minimapa
            self._draw_minimap()
        if self.training_opponent_pool:
            self.current_opponent_idx = (self.current_opponent_idx + 1) % len(self.training_opponent_pool)
    
    def _clear_experience_buffer(self):
        """Limpa buffer de experiência"""
        self.experience_buffer = {
            'obs': [], 'actions': [], 'rewards': [],
            'values': [], 'log_probs': [], 'dones': []
        }
    
    def _reset_training_episode(self):
        """Reseta para novo episódio de treinamento"""
        self.entities.clear()
        self.blue_team.clear()
        self.red_team.clear()
        self.game_over = False
        
        # Posições aleatórias
        if self.current_map:
            map_w, map_h = self.current_map.width, self.current_map.height
        else:
            map_w, map_h = SCREEN_WIDTH, SCREEN_HEIGHT
        
        margin = 150
        agent_x = random.randint(margin, map_w // 2 - 50)
        agent_y = random.randint(margin, map_h - margin)
        opponent_x = random.randint(map_w // 2 + 50, map_w - margin)
        opponent_y = random.randint(margin, map_h - margin)
        
        # Criar agente treinando
        agent_class = self.available_classes[self.training_class]
        agent_weapon = self.available_weapons[self.training_weapon]
        
        self.training_entity = ClassRegistry.create(agent_class, agent_x, agent_y, GREEN)
        self.training_entity.set_weapon(agent_weapon)
        self.training_entity.team = "blue"
        self.training_entity.game_entities = self.entities
        self.entities.append(self.training_entity)
        self.blue_team.append(self.training_entity)
        
        # Criar oponente (IA estratégica)
        opponent_class, opponent_weapon = self._get_current_opponent_config()
        
        ai_controller = StrategicAI(
            class_name=opponent_class.capitalize(),
            weapon_name=opponent_weapon.capitalize()
        )
        
        self.training_opponent = ClassRegistry.create(opponent_class, opponent_x, opponent_y, RED)
        self.training_opponent.set_weapon(opponent_weapon)
        self.training_opponent.team = "red"
        self.training_opponent.set_controller(ai_controller)
        self.training_opponent.game_entities = self.entities
        self.entities.append(self.training_opponent)
        self.red_team.append(self.training_opponent)
        
        ai_controller.set_targets([self.training_entity])
        
        # Resetar estado do episódio
        self.training_step = 0
        self.training_episode_reward = 0
        self._prev_agent_health = self.training_entity.health
        self._prev_opponent_health = self.training_opponent.health
    
    def _update_training(self, dt: float):
        """Atualiza treinamento"""
        if self.training_paused:
            return
        
        # Iniciar treinamento se ainda não iniciou
        if not self.training_active:
            self._start_training()
            return
        
        # Velocidade de treinamento
        steps_per_frame = self.training_speed if self.training_speed > 0 else 10
        
        for _ in range(steps_per_frame):
            if self.game_over:
                break
            
            effective_dt = 1/60
            
            # Guardar vida anterior
            agent_health_before = self.training_entity.health if self.training_entity else 0
            opponent_health_before = self.training_opponent.health if self.training_opponent else 0
            
            # === PASSO DA REDE NEURAL ===
            obs = self._get_observation()
            action, value, log_prob = self._get_neural_action(obs)
            self._apply_neural_action(action)
            
            # Atualizar entidades
            for entity in self.entities:
                if entity.is_alive():
                    entity.update(effective_dt)
            
            # Verificar armadilhas do Trapper
            for entity in self.entities:
                if entity.is_alive() and hasattr(entity, 'check_traps') and hasattr(entity, 'traps'):
                    # Pegar inimigos (entidades do time oposto)
                    enemies = [e for e in self.entities if e.is_alive() and e.team != entity.team]
                    entity.check_traps(enemies)
            
            # Física
            self.physics.handle_collisions(self.entities)
            
            if self.current_map:
                arena_rect = pygame.Rect(0, 0, self.current_map.width, self.current_map.height)
                for entity in self.entities:
                    if entity.is_alive():
                        if self.obstacle_manager:
                            self.obstacle_manager.resolve_collision(entity, effective_dt)
                        self.physics.constrain_to_arena(entity, arena_rect)
                self.physics.check_projectiles_arena_collision(self.entities, arena_rect)
            
            self.training_step += 1
            self.training_step_count += 1
            
            # Calcular reward
            reward = self._calculate_training_reward()
            self.training_episode_reward += reward
            
            # Verificar fim de episódio
            terminated = False
            alive = [e for e in self.entities if e.is_alive()]
            if len(alive) <= 1 or self.training_step > 3000:
                self.game_over = True
                terminated = True
            
            # Armazenar experiência
            if TORCH_AVAILABLE and self.neural_network:
                self.experience_buffer['obs'].append(obs)
                self.experience_buffer['actions'].append(action)
                self.experience_buffer['rewards'].append(reward)
                self.experience_buffer['values'].append(value)
                self.experience_buffer['log_probs'].append(log_prob)
                self.experience_buffer['dones'].append(terminated)
            
            # Atualizar rede neural periodicamente
            if self.training_step_count >= self.ppo_update_freq and TORCH_AVAILABLE and self.neural_network:
                self._update_neural_network()
                self.training_step_count = 0
            
            if self.game_over:
                self._end_training_episode()
                break
    
    def _get_observation(self) -> np.ndarray:
        """Retorna observação normalizada para a rede neural"""
        if not self.training_entity or not self.training_opponent:
            return np.zeros(self.nn_obs_size, dtype=np.float32)
        
        agent = self.training_entity
        opponent = self.training_opponent
        agent_stats = agent.stats_manager.get_stats()
        opponent_stats = opponent.stats_manager.get_stats()
        
        map_w = self.current_map.width if self.current_map else SCREEN_WIDTH
        map_h = self.current_map.height if self.current_map else SCREEN_HEIGHT
        
        obs = []
        
        # Estado do agente (9 valores)
        obs.extend([
            agent.x / map_w,
            agent.y / map_h,
            agent.vx / 20,
            agent.vy / 20,
            agent.facing_angle / math.pi,
            agent.health / agent_stats.max_health,
            1.0 if agent.invulnerable_time > 0 else 0.0,
            1.0 if agent.weapon and agent.weapon.can_attack else 0.0,
            1.0 if agent.ability_cooldown <= 0 else 0.0
        ])
        
        # Estado do oponente relativo (8 valores)
        rel_x = (opponent.x - agent.x) / map_w
        rel_y = (opponent.y - agent.y) / map_h
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
        border_x = min(agent.x, map_w - agent.x) / (map_w / 2)
        border_y = min(agent.y, map_h - agent.y) / (map_h / 2)
        obs.extend([border_x, border_y])
        
        return np.array(obs, dtype=np.float32)
    
    def _get_neural_action(self, obs: np.ndarray):
        """Obtém ação da rede neural"""
        if not TORCH_AVAILABLE or self.neural_network is None:
            return self._get_simple_action(), 0.0, 0.0
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_mean, action_std, value = self.neural_network(obs_tensor)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            return action.squeeze().numpy(), value.item(), log_prob.item()
    
    def _get_simple_action(self) -> np.ndarray:
        """Ação simples de fallback"""
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
        """Aplica ação à entidade"""
        if not self.training_entity or not self.training_entity.is_alive():
            return
        
        move_x = float(np.clip(action[0], -1, 1))
        move_y = float(np.clip(action[1], -1, 1))
        
        if abs(move_x) > 0.1 or abs(move_y) > 0.1:
            self.training_entity.move(move_x, move_y)
        else:
            self.training_entity.moving = False
        
        self._tried_attack_this_frame = False
        
        if action[2] > 0.5:
            self._tried_attack_this_frame = True
            self.training_entity.attack()
        
        if action[3] > 0.5:
            self.training_entity.use_ability()
    
    def _calculate_training_reward(self) -> float:
        """Calcula recompensa do passo"""
        reward = 0.0
        
        # Verificar se entidades existem
        if not self.training_entity or not self.training_entity.is_alive():
            return reward
        
        # Dano causado
        damage_dealt = 0
        if self.training_opponent and self.training_opponent.is_alive():
            damage_dealt = self._prev_opponent_health - self.training_opponent.health
            if damage_dealt > 0:
                reward += damage_dealt * 0.1
            self._prev_opponent_health = self.training_opponent.health
        
        # Penalidade por ataque no vazio
        if self._tried_attack_this_frame and damage_dealt <= 0:
            reward -= 0.02
        
        # Dano recebido
        damage_taken = self._prev_agent_health - self.training_entity.health
        if damage_taken > 0:
            reward -= damage_taken * 0.05
        self._prev_agent_health = self.training_entity.health
        
        self._tried_attack_this_frame = False
        return reward
    
    def _end_training_episode(self):
        """Finaliza episódio de treinamento"""
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
        
        # Adicionar ao histórico para gráficos
        self.reward_history.append(self.training_episode_reward)
        total_games = self.training_stats['wins'] + self.training_stats['losses']
        if total_games > 0:
            current_win_rate = self.training_stats['wins'] / total_games * 100
            self.win_rate_history.append(current_win_rate)
        
        # Verificar se terminou treinamento
        if self.training_stats['episode'] >= self.training_stats['total_episodes']:
            self._stop_training()
            return
        
        # Auto-save periódico
        if self.training_stats['episode'] % self.training_save_freq == 0:
            self._save_training_model()
        
        # Rotacionar oponente
        if self.training_multi_opponent:
            if self.training_stats['episode'] % self.episodes_per_opponent == 0:
                self._rotate_opponent()
        
        # Resetar para próximo episódio
        self._reset_training_episode()
    
    def _update_neural_network(self):
        """Atualiza a rede neural usando PPO"""
        if not TORCH_AVAILABLE or self.neural_network is None:
            return
        
        if len(self.experience_buffer['obs']) < self.ppo_batch_size:
            return
        
        obs_tensor = torch.FloatTensor(np.array(self.experience_buffer['obs']))
        action_tensor = torch.FloatTensor(np.array(self.experience_buffer['actions']))
        old_log_probs = torch.FloatTensor(np.array(self.experience_buffer['log_probs']))
        rewards = np.array(self.experience_buffer['rewards'])
        values = np.array(self.experience_buffer['values'])
        dones = np.array(self.experience_buffer['dones'])
        
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)
        
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
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
                
                batch_obs = obs_tensor[batch_indices]
                batch_actions = action_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                values_pred, log_probs, entropy = self.neural_network.evaluate_actions(
                    batch_obs, batch_actions
                )
                
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.ppo_clip_epsilon, 1 + self.ppo_clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values_pred, batch_returns)
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                self.nn_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.neural_network.parameters(), 0.5)
                self.nn_optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        if num_updates > 0:
            self.training_stats['policy_loss'] = total_policy_loss / num_updates
            self.training_stats['value_loss'] = total_value_loss / num_updates
            self.training_stats['entropy'] = total_entropy / num_updates
            self.training_stats['learning_updates'] += 1
        
        self._clear_experience_buffer()
    
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
    
    def _stop_training(self):
        """Para o treinamento"""
        self.training_active = False
        elapsed = time.time() - self.training_stats['training_time']
        self.training_stats['training_time'] = elapsed
        self._save_training_model()
        self.message = "✅ Treinamento concluído!"
        self.message_timer = 3.0
    
    def _save_training_model(self):
        """Salva o modelo atual"""
        agent_class = self.available_classes[self.training_class]
        agent_weapon = self.available_weapons[self.training_weapon]
        
        opp_suffix = "multi" if self.training_multi_opponent else "single"
        
        os.makedirs("models", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_name = f"model_{agent_class}_{agent_weapon}_vs_{opp_suffix}_ep{self.training_stats['episode']}_{timestamp}"
        
        # Salvar pesos da rede neural
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
        
        # Salvar estatísticas
        elapsed = time.time() - self.training_stats['training_time'] if isinstance(self.training_stats['training_time'], float) and self.training_stats['training_time'] > 1000000 else self.training_stats['training_time']
        
        model_data = {
            'agent_class': agent_class,
            'agent_weapon': agent_weapon,
            'multi_opponent': self.training_multi_opponent,
            'episode': self.training_stats['episode'],
            'total_episodes': self.training_stats['total_episodes'],
            'wins': self.training_stats['wins'],
            'losses': self.training_stats['losses'],
            'win_rate': self.training_stats['wins'] / max(1, self.training_stats['episode']) * 100,
            'avg_reward': self.training_stats['avg_reward'],
            'best_reward': self.training_stats['best_reward'],
            'training_time': elapsed,
            'has_neural_weights': has_neural_weights,
            'weights_file': f"{base_name}.pth" if has_neural_weights else None,
        }
        
        json_path = f"models/{base_name}.json"
        with open(json_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        self.message = f"✅ Modelo salvo: {base_name}"
        self.message_timer = 2.0
        
        # Atualizar lista de modelos
        self.available_models = self._scan_models()
    
    def _draw_training(self):
        """Desenha a tela de treinamento"""
        if not self.current_map:
            screen.fill(DARK_GRAY)
            return
        
        # Desenhar jogo se renderização ativa
        if self.training_render:
            # Atualizar câmera
            self._update_camera()
            
            # Desenhar mundo
            self._draw_game_world()
        else:
            screen.fill(self.current_map.bg_color)
            msg = fonts['large'].render("RENDERIZAÇÃO DESATIVADA (V para ativar)", True, WHITE)
            rect = msg.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            screen.blit(msg, rect)
        
        # Painel de estatísticas (direita)
        self._draw_training_stats()
        
        # Gráfico de rewards (esquerda inferior)
        self._draw_reward_graph()
        
        # Info do modelo sendo treinado (topo)
        agent_class = self.available_classes[self.training_class]
        agent_weapon = self.available_weapons[self.training_weapon]
        model_text = f"Treinando: {agent_class} + {agent_weapon}"
        model_surf = fonts['medium'].render(model_text, True, GREEN)
        screen.blit(model_surf, (10, 10))
        
        # Oponente atual
        opp_class, opp_weapon = self._get_current_opponent_config()
        opp_text = f"vs {opp_class} + {opp_weapon}"
        opp_surf = fonts['small'].render(opp_text, True, RED)
        screen.blit(opp_surf, (10, 40))
        
        # Instruções
        inst = fonts['small'].render(
            "ESPAÇO: Pausar | V: Render | +/-: Velocidade | Z/X: Zoom | C: Alvo | ESC: Voltar",
            True, WHITE
        )
        screen.blit(inst, (10, SCREEN_HEIGHT - 30))
    
    def _draw_training_stats(self):
        """Desenha painel de estatísticas do treinamento"""
        panel_w = 280
        panel_h = 280
        panel_x = SCREEN_WIDTH - panel_w - 10
        panel_y = 10
        
        # Background
        panel_surf = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel_surf.fill((0, 0, 0, 200))
        screen.blit(panel_surf, (panel_x, panel_y))
        pygame.draw.rect(screen, YELLOW, (panel_x, panel_y, panel_w, panel_h), 2)
        
        # Título
        title = fonts['medium'].render("📊 ESTATÍSTICAS", True, YELLOW)
        screen.blit(title, (panel_x + 10, panel_y + 8))
        
        y = panel_y + 45
        
        # Progresso
        progress = self.training_stats['episode'] / max(1, self.training_stats['total_episodes'])
        progress_text = f"Progresso: {progress * 100:.1f}%"
        progress_surf = fonts['small'].render(progress_text, True, WHITE)
        screen.blit(progress_surf, (panel_x + 10, y))
        
        # Barra de progresso
        bar_y = y + 20
        bar_w = panel_w - 20
        bar_h = 12
        pygame.draw.rect(screen, (40, 40, 50), (panel_x + 10, bar_y, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            pygame.draw.rect(screen, GREEN, (panel_x + 10, bar_y, fill_w, bar_h), border_radius=3)
        
        y = bar_y + 25
        
        # Estatísticas
        win_rate = self.training_stats['wins'] / max(1, self.training_stats['episode']) * 100
        
        stats_lines = [
            (f"Episódio: {self.training_stats['episode']}", WHITE),
            (f"Vitórias: {self.training_stats['wins']}", GREEN),
            (f"Derrotas: {self.training_stats['losses']}", RED),
            (f"Win Rate: {win_rate:.1f}%", YELLOW if win_rate > 50 else (200, 100, 100)),
            (f"Reward: {self.training_stats['avg_reward']:.2f}", WHITE),
            (f"Best: {self.training_stats['best_reward']:.2f}", (100, 200, 255)),
            ("", WHITE),  # Espaço
            (f"Velocidade: {'MAX' if self.training_speed == 0 else f'{self.training_speed}x'}", GRAY),
            (f"Render: {'ON' if self.training_render else 'OFF'}", GRAY),
            (f"{'⏸️ PAUSADO' if self.training_paused else '▶️ RODANDO'}", YELLOW if self.training_paused else GREEN),
        ]
        
        for text, color in stats_lines:
            if text:
                surf = fonts['small'].render(text, True, color)
                screen.blit(surf, (panel_x + 10, y))
            y += 18
    
    def _draw_reward_graph(self):
        """Desenha gráfico de rewards"""
        if len(self.reward_history) < 2:
            return
        
        graph_w = 350
        graph_h = 150
        graph_x = 10
        graph_y = SCREEN_HEIGHT - graph_h - 50
        
        # Background
        graph_surf = pygame.Surface((graph_w, graph_h), pygame.SRCALPHA)
        graph_surf.fill((0, 0, 0, 180))
        screen.blit(graph_surf, (graph_x, graph_y))
        pygame.draw.rect(screen, GRAY, (graph_x, graph_y, graph_w, graph_h), 1)
        
        # Título
        title = fonts['small'].render("Reward por Episódio", True, WHITE)
        screen.blit(title, (graph_x + 5, graph_y + 3))
        
        # Calcular escala
        rewards = list(self.reward_history)
        min_r = min(rewards)
        max_r = max(rewards)
        range_r = max_r - min_r if max_r != min_r else 1
        
        # Linha de zero
        zero_y = graph_y + graph_h - 20 - int((0 - min_r) / range_r * (graph_h - 40))
        if graph_y + 20 < zero_y < graph_y + graph_h - 20:
            pygame.draw.line(screen, (80, 80, 80), (graph_x, zero_y), (graph_x + graph_w, zero_y), 1)
        
        # Desenhar linha do gráfico
        if len(rewards) > 1:
            points = []
            for i, reward in enumerate(rewards):
                x = graph_x + 5 + int(i / len(rewards) * (graph_w - 10))
                y = graph_y + graph_h - 20 - int((reward - min_r) / range_r * (graph_h - 40))
                points.append((x, y))
            
            if len(points) >= 2:
                # Linha suavizada (média móvel visual)
                pygame.draw.lines(screen, GREEN, False, points, 2)
        
        # Labels
        max_label = fonts['small'].render(f"{max_r:.1f}", True, (100, 200, 100))
        screen.blit(max_label, (graph_x + graph_w - 50, graph_y + 20))
        
        min_label = fonts['small'].render(f"{min_r:.1f}", True, (200, 100, 100))
        screen.blit(min_label, (graph_x + graph_w - 50, graph_y + graph_h - 35))
    
    # ==========================================================================
    # LOOP PRINCIPAL
    # ==========================================================================
    
    def run(self):
        """Loop principal do jogo"""
        while self.running:
            dt = clock.tick(FPS) / 1000.0
            
            self.handle_events()
            self.update(dt)
            self.draw()
        
        pygame.quit()


# ============================================================================
# MAIN
# ============================================================================

def main():
    game = Game()
    game.run()


if __name__ == "__main__":
    main()
