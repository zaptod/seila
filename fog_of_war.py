"""
Sistema de Fog of War e Visão
=============================
Gerencia a visibilidade de entidades baseada em campo de visão por classe.
Inclui sistema de obstáculos que bloqueiam visão e movimento.
"""

import pygame
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING
import random

if TYPE_CHECKING:
    from entities import Entity


# ============================================================================
# CONFIGURAÇÃO DE VISÃO POR CLASSE
# ============================================================================

# Raio de visão base por classe (pode ser modificado por buffs)
VISION_RADIUS_BY_CLASS = {
    # DPS Melee - visão média
    'warrior': 280,
    'berserker': 250,
    'assassin': 320,  # Assassin tem boa percepção
    'lancer': 300,
    
    # Tanks - visão menor (focados em combate próximo)
    'tank': 220,
    'guardian': 240,
    
    # Support - visão média-alta (precisam ver aliados)
    'cleric': 350,
    'enchanter': 340,
    
    # Control - visão boa (precisam ver inimigos para CC)
    'controller': 320,
    'trapper': 380,  # Trapper precisa ver onde colocar armadilhas
    
    # Ranged - visão muito alta
    'ranger': 450,  # Arqueiro tem a maior visão
}

# Ângulo do cone de visão (em graus) - alguns têm visão periférica melhor
VISION_ANGLE_BY_CLASS = {
    'warrior': 180,
    'berserker': 160,  # Berserker focado na frente
    'assassin': 220,  # Assassin percebe mais ao redor
    'tank': 200,
    'lancer': 170,
    'cleric': 270,  # Suportes têm boa visão periférica
    'guardian': 240,
    'controller': 200,
    'ranger': 160,  # Ranger focado na frente, mas longe
    'enchanter': 260,
    'trapper': 240,
}

# Raio de visão padrão para classes não mapeadas
DEFAULT_VISION_RADIUS = 300
DEFAULT_VISION_ANGLE = 180


def get_vision_radius(class_id: str) -> float:
    """Retorna o raio de visão para uma classe"""
    return VISION_RADIUS_BY_CLASS.get(class_id, DEFAULT_VISION_RADIUS)


def get_vision_angle(class_id: str) -> float:
    """Retorna o ângulo de visão (em radianos) para uma classe"""
    angle_deg = VISION_ANGLE_BY_CLASS.get(class_id, DEFAULT_VISION_ANGLE)
    return math.radians(angle_deg)


# ============================================================================
# OBSTÁCULOS
# ============================================================================

@dataclass
class Obstacle:
    """Um obstáculo no mapa que bloqueia movimento e/ou visão"""
    x: float
    y: float
    width: float
    height: float
    blocks_movement: bool = True
    blocks_vision: bool = True
    destructible: bool = False
    health: float = 100
    color: Tuple[int, int, int] = (80, 80, 90)
    obstacle_type: str = "rock"  # rock, wall, tree, pillar, etc.
    
    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Verifica se um ponto está dentro do obstáculo"""
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)
    
    def collides_with_circle(self, cx: float, cy: float, radius: float) -> bool:
        """Verifica colisão com um círculo (entidade)"""
        # Encontrar o ponto mais próximo no retângulo
        closest_x = max(self.x, min(cx, self.x + self.width))
        closest_y = max(self.y, min(cy, self.y + self.height))
        
        # Calcular distância
        dx = cx - closest_x
        dy = cy - closest_y
        distance = math.sqrt(dx * dx + dy * dy)
        
        return distance < radius
    
    def get_collision_normal(self, cx: float, cy: float) -> Tuple[float, float]:
        """Retorna o vetor normal da colisão"""
        # Encontrar o ponto mais próximo
        closest_x = max(self.x, min(cx, self.x + self.width))
        closest_y = max(self.y, min(cy, self.y + self.height))
        
        dx = cx - closest_x
        dy = cy - closest_y
        dist = math.sqrt(dx * dx + dy * dy)
        
        if dist == 0:
            return (0, -1)  # Default para cima
        
        return (dx / dist, dy / dist)
    
    def draw(self, screen: pygame.Surface, camera_offset: Tuple[float, float] = (0, 0)):
        """Desenha o obstáculo"""
        draw_x = int(self.x - camera_offset[0])
        draw_y = int(self.y - camera_offset[1])
        
        rect = pygame.Rect(draw_x, draw_y, int(self.width), int(self.height))
        
        if self.obstacle_type == "rock":
            # Rocha - cinza com bordas
            pygame.draw.rect(screen, self.color, rect)
            pygame.draw.rect(screen, (60, 60, 70), rect, 2)
            # Detalhes
            pygame.draw.line(screen, (100, 100, 110), 
                           (draw_x + 5, draw_y + 5),
                           (draw_x + self.width - 5, draw_y + self.height // 2), 1)
        
        elif self.obstacle_type == "wall":
            # Parede - marrom/bege
            pygame.draw.rect(screen, self.color, rect)
            pygame.draw.rect(screen, (100, 80, 60), rect, 3)
            # Tijolos
            brick_h = self.height // 3
            for i in range(3):
                y_offset = i * brick_h
                pygame.draw.line(screen, (90, 70, 50),
                               (draw_x, draw_y + y_offset),
                               (draw_x + self.width, draw_y + y_offset), 1)
        
        elif self.obstacle_type == "tree":
            # Árvore - tronco + copa
            trunk_w = self.width // 3
            trunk_h = self.height // 2
            trunk_x = draw_x + self.width // 2 - trunk_w // 2
            trunk_y = draw_y + self.height - trunk_h
            
            pygame.draw.rect(screen, (80, 50, 30), 
                           (trunk_x, trunk_y, trunk_w, trunk_h))
            # Copa
            pygame.draw.circle(screen, (40, 100, 40),
                             (draw_x + int(self.width // 2), draw_y + int(self.height // 3)),
                             int(self.width // 2))
        
        elif self.obstacle_type == "pillar":
            # Pilar - circular
            center = (draw_x + int(self.width // 2), draw_y + int(self.height // 2))
            radius = int(min(self.width, self.height) // 2)
            pygame.draw.circle(screen, self.color, center, radius)
            pygame.draw.circle(screen, (100, 100, 110), center, radius, 2)
        
        elif self.obstacle_type == "bush":
            # Arbusto - só bloqueia visão, não movimento
            center = (draw_x + int(self.width // 2), draw_y + int(self.height // 2))
            pygame.draw.ellipse(screen, (30, 80, 30), rect)
            pygame.draw.ellipse(screen, (40, 100, 40), rect, 2)
        
        else:
            # Padrão
            pygame.draw.rect(screen, self.color, rect)
            pygame.draw.rect(screen, (100, 100, 110), rect, 2)


class ObstacleManager:
    """Gerencia todos os obstáculos do mapa"""
    
    def __init__(self):
        self.obstacles: List[Obstacle] = []
    
    def clear(self):
        """Remove todos os obstáculos"""
        self.obstacles.clear()
    
    def add(self, obstacle: Obstacle):
        """Adiciona um obstáculo"""
        self.obstacles.append(obstacle)
    
    def generate_for_map(self, map_width: int, map_height: int, 
                         map_type: str = "large_arena", 
                         border: int = 100,
                         density: float = 0.3):
        """
        Gera obstáculos proceduralmente para o mapa.
        density: 0.0 a 1.0, quanto maior mais obstáculos
        """
        self.clear()
        
        # Área jogável
        play_area_x = border
        play_area_y = border
        play_area_w = map_width - 2 * border
        play_area_h = map_height - 2 * border
        
        # Spawns dos times (evitar colocar obstáculos aqui)
        spawn_radius = 200
        blue_spawn = (border + 150, map_height // 2)
        red_spawn = (map_width - border - 150, map_height // 2)
        
        # Configuração baseada no tipo de mapa
        if map_type == "large_arena":
            self._generate_arena_obstacles(play_area_x, play_area_y, play_area_w, play_area_h,
                                          blue_spawn, red_spawn, spawn_radius, density)
        elif map_type == "forest":
            self._generate_forest_obstacles(play_area_x, play_area_y, play_area_w, play_area_h,
                                           blue_spawn, red_spawn, spawn_radius, density)
        elif map_type == "ruins":
            self._generate_ruins_obstacles(play_area_x, play_area_y, play_area_w, play_area_h,
                                          blue_spawn, red_spawn, spawn_radius, density)
        elif map_type == "canyon":
            self._generate_canyon_obstacles(play_area_x, play_area_y, play_area_w, play_area_h,
                                           blue_spawn, red_spawn, spawn_radius, density)
    
    def _is_near_spawn(self, x: float, y: float, w: float, h: float,
                       spawns: List[Tuple[float, float]], spawn_radius: float) -> bool:
        """Verifica se um retângulo está perto de uma área de spawn"""
        center_x = x + w / 2
        center_y = y + h / 2
        
        for sx, sy in spawns:
            dist = math.sqrt((center_x - sx) ** 2 + (center_y - sy) ** 2)
            if dist < spawn_radius + max(w, h):
                return True
        return False
    
    def _generate_arena_obstacles(self, x, y, w, h, blue_spawn, red_spawn, spawn_r, density):
        """Gera obstáculos para arena grande"""
        random.seed(100)  # Seed fixa para consistência
        
        spawns = [blue_spawn, red_spawn]
        
        # Pilares centrais em padrão simétrico
        center_x = x + w // 2
        center_y = y + h // 2
        
        # Pilar central
        pillar_size = 60
        self.add(Obstacle(
            center_x - pillar_size // 2, center_y - pillar_size // 2,
            pillar_size, pillar_size,
            obstacle_type="pillar", color=(90, 90, 100)
        ))
        
        # Pilares em círculo ao redor do centro
        for i in range(6):
            angle = i * math.pi / 3
            dist = 300
            px = center_x + math.cos(angle) * dist - pillar_size // 2
            py = center_y + math.sin(angle) * dist - pillar_size // 2
            
            if not self._is_near_spawn(px, py, pillar_size, pillar_size, spawns, spawn_r):
                self.add(Obstacle(
                    px, py, pillar_size, pillar_size,
                    obstacle_type="pillar", color=(80, 80, 90)
                ))
        
        # Rochas espalhadas
        num_rocks = int(30 * density)
        for _ in range(num_rocks):
            rw = random.randint(40, 100)
            rh = random.randint(40, 80)
            rx = random.randint(x + 50, x + w - rw - 50)
            ry = random.randint(y + 50, y + h - rh - 50)
            
            if not self._is_near_spawn(rx, ry, rw, rh, spawns, spawn_r):
                self.add(Obstacle(
                    rx, ry, rw, rh,
                    obstacle_type="rock", color=(70 + random.randint(-10, 10), 
                                                  70 + random.randint(-10, 10),
                                                  80 + random.randint(-10, 10))
                ))
        
        # Paredes curtas para cobertura
        num_walls = int(12 * density)
        for _ in range(num_walls):
            ww = random.randint(80, 150)
            wh = random.randint(20, 40)
            
            # Alternar orientação
            if random.random() > 0.5:
                ww, wh = wh, ww
            
            wx = random.randint(x + 100, x + w - ww - 100)
            wy = random.randint(y + 100, y + h - wh - 100)
            
            if not self._is_near_spawn(wx, wy, ww, wh, spawns, spawn_r):
                self.add(Obstacle(
                    wx, wy, ww, wh,
                    obstacle_type="wall", color=(120, 100, 80)
                ))
    
    def _generate_forest_obstacles(self, x, y, w, h, blue_spawn, red_spawn, spawn_r, density):
        """Gera obstáculos de floresta (muitas árvores)"""
        random.seed(101)
        spawns = [blue_spawn, red_spawn]
        
        # Muitas árvores
        num_trees = int(60 * density)
        for _ in range(num_trees):
            tw = random.randint(50, 80)
            th = random.randint(60, 100)
            tx = random.randint(x + 30, x + w - tw - 30)
            ty = random.randint(y + 30, y + h - th - 30)
            
            if not self._is_near_spawn(tx, ty, tw, th, spawns, spawn_r):
                self.add(Obstacle(
                    tx, ty, tw, th,
                    obstacle_type="tree", color=(50, 90, 40)
                ))
        
        # Arbustos (só bloqueiam visão)
        num_bushes = int(40 * density)
        for _ in range(num_bushes):
            bw = random.randint(40, 70)
            bh = random.randint(30, 50)
            bx = random.randint(x + 20, x + w - bw - 20)
            by = random.randint(y + 20, y + h - bh - 20)
            
            if not self._is_near_spawn(bx, by, bw, bh, spawns, spawn_r):
                self.add(Obstacle(
                    bx, by, bw, bh,
                    blocks_movement=False,  # Pode atravessar
                    blocks_vision=True,
                    obstacle_type="bush", color=(30, 70, 30)
                ))
    
    def _generate_ruins_obstacles(self, x, y, w, h, blue_spawn, red_spawn, spawn_r, density):
        """Gera obstáculos de ruínas (paredes quebradas)"""
        random.seed(102)
        spawns = [blue_spawn, red_spawn]
        
        # Paredes em ruínas
        num_walls = int(25 * density)
        for _ in range(num_walls):
            ww = random.randint(60, 200)
            wh = random.randint(15, 35)
            
            if random.random() > 0.5:
                ww, wh = wh, ww
            
            wx = random.randint(x + 50, x + w - ww - 50)
            wy = random.randint(y + 50, y + h - wh - 50)
            
            if not self._is_near_spawn(wx, wy, ww, wh, spawns, spawn_r):
                self.add(Obstacle(
                    wx, wy, ww, wh,
                    obstacle_type="wall", color=(100, 90, 80)
                ))
        
        # Pilares caídos
        num_pillars = int(15 * density)
        for _ in range(num_pillars):
            pw = random.randint(40, 60)
            ph = random.randint(40, 60)
            px = random.randint(x + 50, x + w - pw - 50)
            py = random.randint(y + 50, y + h - ph - 50)
            
            if not self._is_near_spawn(px, py, pw, ph, spawns, spawn_r):
                self.add(Obstacle(
                    px, py, pw, ph,
                    obstacle_type="pillar", color=(110, 100, 90)
                ))
    
    def _generate_canyon_obstacles(self, x, y, w, h, blue_spawn, red_spawn, spawn_r, density):
        """Gera obstáculos de canyon (corredores com rochas)"""
        random.seed(103)
        spawns = [blue_spawn, red_spawn]
        
        center_y = y + h // 2
        
        # Criar corredores com paredes de rocha
        corridor_height = h // 3
        
        # Parede superior
        for i in range(0, w, 150):
            rw = random.randint(100, 180)
            rh = random.randint(80, 150)
            rx = x + i + random.randint(-20, 20)
            ry = y + random.randint(20, 80)
            
            if not self._is_near_spawn(rx, ry, rw, rh, spawns, spawn_r):
                self.add(Obstacle(
                    rx, ry, rw, rh,
                    obstacle_type="rock", color=(90, 80, 70)
                ))
        
        # Parede inferior
        for i in range(0, w, 150):
            rw = random.randint(100, 180)
            rh = random.randint(80, 150)
            rx = x + i + random.randint(-20, 20)
            ry = y + h - rh - random.randint(20, 80)
            
            if not self._is_near_spawn(rx, ry, rw, rh, spawns, spawn_r):
                self.add(Obstacle(
                    rx, ry, rw, rh,
                    obstacle_type="rock", color=(90, 80, 70)
                ))
        
        # Rochas no meio para criar cobertura
        num_center_rocks = int(20 * density)
        for _ in range(num_center_rocks):
            rw = random.randint(50, 100)
            rh = random.randint(40, 80)
            rx = random.randint(x + 200, x + w - rw - 200)
            ry = center_y + random.randint(-corridor_height // 2, corridor_height // 2) - rh // 2
            
            if not self._is_near_spawn(rx, ry, rw, rh, spawns, spawn_r):
                self.add(Obstacle(
                    rx, ry, rw, rh,
                    obstacle_type="rock", color=(80, 70, 65)
                ))
    
    def check_collision(self, x: float, y: float, radius: float) -> Optional[Obstacle]:
        """Verifica colisão com obstáculos que bloqueiam movimento"""
        for obs in self.obstacles:
            if obs.blocks_movement and obs.collides_with_circle(x, y, radius):
                return obs
        return None
    
    def resolve_collision(self, entity: 'Entity', dt: float) -> bool:
        """
        Resolve colisão entre entidade e obstáculos.
        Retorna True se houve colisão.
        """
        had_collision = False
        
        for obs in self.obstacles:
            if not obs.blocks_movement:
                continue
            
            if obs.collides_with_circle(entity.x, entity.y, entity.radius):
                had_collision = True
                
                # Obter normal da colisão
                nx, ny = obs.get_collision_normal(entity.x, entity.y)
                
                # Empurrar entidade para fora
                overlap = entity.radius - math.sqrt(
                    (entity.x - max(obs.x, min(entity.x, obs.x + obs.width))) ** 2 +
                    (entity.y - max(obs.y, min(entity.y, obs.y + obs.height))) ** 2
                )
                
                if overlap > 0:
                    entity.x += nx * (overlap + 1)
                    entity.y += ny * (overlap + 1)
                
                # Anular velocidade na direção do obstáculo
                dot = entity.vx * (-nx) + entity.vy * (-ny)
                if dot > 0:
                    entity.vx += nx * dot * 0.5
                    entity.vy += ny * dot * 0.5
        
        return had_collision
    
    def blocks_line_of_sight(self, x1: float, y1: float, x2: float, y2: float) -> bool:
        """Verifica se a linha de visão está bloqueada por obstáculos"""
        for obs in self.obstacles:
            if not obs.blocks_vision:
                continue
            
            # Verificar interseção linha-retângulo
            if self._line_intersects_rect(x1, y1, x2, y2, obs):
                return True
        
        return False
    
    def _line_intersects_rect(self, x1: float, y1: float, x2: float, y2: float, 
                               obs: Obstacle) -> bool:
        """Verifica se uma linha intersecta um retângulo"""
        # Verificar se ambos os pontos estão do mesmo lado de cada borda
        rect_x = obs.x
        rect_y = obs.y
        rect_w = obs.width
        rect_h = obs.height
        
        # Usar algoritmo de Cohen-Sutherland simplificado
        # Verificar interseção com cada aresta
        edges = [
            (rect_x, rect_y, rect_x + rect_w, rect_y),  # Top
            (rect_x, rect_y + rect_h, rect_x + rect_w, rect_y + rect_h),  # Bottom
            (rect_x, rect_y, rect_x, rect_y + rect_h),  # Left
            (rect_x + rect_w, rect_y, rect_x + rect_w, rect_y + rect_h),  # Right
        ]
        
        for ex1, ey1, ex2, ey2 in edges:
            if self._lines_intersect(x1, y1, x2, y2, ex1, ey1, ex2, ey2):
                return True
        
        # Verificar se a linha está completamente dentro do retângulo
        if (rect_x <= x1 <= rect_x + rect_w and rect_y <= y1 <= rect_y + rect_h):
            return True
        if (rect_x <= x2 <= rect_x + rect_w and rect_y <= y2 <= rect_y + rect_h):
            return True
        
        return False
    
    def _lines_intersect(self, x1, y1, x2, y2, x3, y3, x4, y4) -> bool:
        """Verifica se duas linhas se intersectam"""
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 0.0001:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def draw(self, screen: pygame.Surface, camera_offset: Tuple[float, float] = (0, 0)):
        """Desenha todos os obstáculos"""
        for obs in self.obstacles:
            obs.draw(screen, camera_offset)


# ============================================================================
# SISTEMA DE FOG OF WAR
# ============================================================================

class FogOfWar:
    """Sistema de Fog of War que limita a visão das entidades"""
    
    def __init__(self, map_width: int, map_height: int):
        self.map_width = map_width
        self.map_height = map_height
        self.obstacle_manager: Optional[ObstacleManager] = None
        
        # Cache de visibilidade (recalculado a cada frame)
        self._visibility_cache: Dict[int, List[Tuple[float, float]]] = {}
        
        # Configuração
        self.enabled = True
        self.fog_color = (20, 20, 30, 200)  # Cor do fog (RGBA)
        self.explored_fog_color = (40, 40, 50, 150)  # Áreas já exploradas
        
        # Áreas exploradas por time
        self._explored_blue: set = set()
        self._explored_red: set = set()
        self.exploration_grid_size = 50  # Tamanho da grid de exploração
    
    def set_obstacle_manager(self, manager: ObstacleManager):
        """Define o gerenciador de obstáculos para verificação de linha de visão"""
        self.obstacle_manager = manager
    
    def is_visible_by(self, viewer: 'Entity', target_x: float, target_y: float) -> bool:
        """
        Verifica se uma posição é visível por uma entidade.
        Considera raio de visão, ângulo de visão e obstáculos.
        """
        if not self.enabled:
            return True
        
        # Distância
        dx = target_x - viewer.x
        dy = target_y - viewer.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Raio de visão da classe
        vision_radius = get_vision_radius(viewer.class_id)
        
        # Buffs podem aumentar visão
        if hasattr(viewer, 'status_effects') and viewer.status_effects:
            if viewer.status_effects.has_effect('vision_buff'):
                vision_radius *= 1.3
        
        if distance > vision_radius:
            return False
        
        # Verificar ângulo de visão
        vision_angle = get_vision_angle(viewer.class_id)
        
        if vision_angle < math.pi * 2 - 0.1:  # Se não for 360 graus
            angle_to_target = math.atan2(dy, dx)
            angle_diff = abs(self._normalize_angle(angle_to_target - viewer.facing_angle))
            
            if angle_diff > vision_angle / 2:
                return False
        
        # Verificar obstáculos bloqueando visão
        if self.obstacle_manager:
            if self.obstacle_manager.blocks_line_of_sight(viewer.x, viewer.y, target_x, target_y):
                return False
        
        return True
    
    def is_entity_visible_by(self, viewer: 'Entity', target: 'Entity') -> bool:
        """Verifica se uma entidade é visível por outra"""
        return self.is_visible_by(viewer, target.x, target.y)
    
    def is_visible_by_team(self, team: str, entities: List['Entity'], 
                           target_x: float, target_y: float) -> bool:
        """Verifica se uma posição é visível por alguma entidade do time"""
        for entity in entities:
            if entity.team == team and entity.is_alive():
                if self.is_visible_by(entity, target_x, target_y):
                    return True
        return False
    
    def _normalize_angle(self, angle: float) -> float:
        """Normaliza ângulo para -pi a pi"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def update_exploration(self, entities: List['Entity']):
        """Atualiza as áreas exploradas pelos times"""
        for entity in entities:
            if not entity.is_alive():
                continue
            
            vision_radius = get_vision_radius(entity.class_id)
            
            # Marcar células da grid como exploradas
            grid_x = int(entity.x // self.exploration_grid_size)
            grid_y = int(entity.y // self.exploration_grid_size)
            cells_radius = int(vision_radius // self.exploration_grid_size) + 1
            
            explored_set = self._explored_blue if entity.team == "blue" else self._explored_red
            
            for dx in range(-cells_radius, cells_radius + 1):
                for dy in range(-cells_radius, cells_radius + 1):
                    cell = (grid_x + dx, grid_y + dy)
                    cell_center_x = (cell[0] + 0.5) * self.exploration_grid_size
                    cell_center_y = (cell[1] + 0.5) * self.exploration_grid_size
                    
                    dist = math.sqrt((cell_center_x - entity.x) ** 2 + 
                                    (cell_center_y - entity.y) ** 2)
                    
                    if dist <= vision_radius:
                        explored_set.add(cell)
    
    def draw_fog(self, screen: pygame.Surface, viewing_team: str,
                 entities: List['Entity'], camera_offset: Tuple[float, float] = (0, 0)):
        """
        Desenha o fog of war para o time especificado.
        Entidades inimigas fora do campo de visão ficam invisíveis.
        """
        if not self.enabled:
            return
        
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Criar surface para o fog
        fog_surface = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
        fog_surface.fill(self.fog_color)
        
        # Desenhar áreas visíveis (transparentes)
        team_entities = [e for e in entities if e.team == viewing_team and e.is_alive()]
        
        for entity in team_entities:
            vision_radius = get_vision_radius(entity.class_id)
            vision_angle = get_vision_angle(entity.class_id)
            
            # Posição na tela
            screen_x = entity.x - camera_offset[0]
            screen_y = entity.y - camera_offset[1]
            
            if vision_angle >= math.pi * 2 - 0.1:
                # Visão 360 graus - desenhar círculo
                pygame.draw.circle(fog_surface, (0, 0, 0, 0),
                                 (int(screen_x), int(screen_y)), int(vision_radius))
            else:
                # Visão em cone
                self._draw_vision_cone(fog_surface, screen_x, screen_y,
                                      vision_radius, vision_angle, entity.facing_angle)
        
        # Aplicar fog
        screen.blit(fog_surface, (0, 0))
    
    def _draw_vision_cone(self, surface: pygame.Surface, x: float, y: float,
                          radius: float, angle: float, facing: float):
        """Desenha um cone de visão transparente"""
        # Criar pontos do cone
        num_points = 20
        half_angle = angle / 2
        
        points = [(int(x), int(y))]
        
        for i in range(num_points + 1):
            point_angle = facing - half_angle + (angle * i / num_points)
            px = x + math.cos(point_angle) * radius
            py = y + math.sin(point_angle) * radius
            points.append((int(px), int(py)))
        
        # Desenhar cone transparente
        if len(points) >= 3:
            pygame.draw.polygon(surface, (0, 0, 0, 0), points)
    
    def get_visible_entities(self, viewer: 'Entity', all_entities: List['Entity']) -> List['Entity']:
        """Retorna lista de entidades visíveis pelo viewer"""
        visible = []
        
        for entity in all_entities:
            if entity == viewer:
                continue
            
            # Aliados sempre visíveis
            if entity.team == viewer.team:
                visible.append(entity)
            # Inimigos só se estiverem no campo de visão
            elif self.is_entity_visible_by(viewer, entity):
                visible.append(entity)
        
        return visible
    
    def get_team_visible_entities(self, team: str, all_entities: List['Entity']) -> List['Entity']:
        """Retorna entidades visíveis por qualquer membro do time"""
        team_members = [e for e in all_entities if e.team == team and e.is_alive()]
        visible = set()
        
        for entity in all_entities:
            # Aliados sempre visíveis
            if entity.team == team:
                visible.add(entity)
                continue
            
            # Verificar se algum aliado vê o inimigo
            for viewer in team_members:
                if self.is_entity_visible_by(viewer, entity):
                    visible.add(entity)
                    break
        
        return list(visible)


# ============================================================================
# SISTEMA DE CÂMERA
# ============================================================================

class Camera:
    """Câmera que segue entidades em mapas grandes"""
    
    def __init__(self, screen_width: int, screen_height: int, 
                 map_width: int, map_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.map_width = map_width
        self.map_height = map_height
        
        # Posição da câmera (centro)
        self.x = map_width // 2
        self.y = map_height // 2
        
        # Suavização do movimento
        self.smoothing = 0.1
        
        # Zoom
        self.zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 2.0
        
        # Alvo da câmera
        self.target: Optional['Entity'] = None
        self.targets: List['Entity'] = []
        
        # Modo de câmera
        self.mode = "follow"  # "follow", "center_on_action", "overview"
    
    def set_target(self, entity: 'Entity'):
        """Define uma entidade para a câmera seguir"""
        self.target = entity
        self.targets = [entity]
        self.mode = "follow"
    
    def set_targets(self, entities: List['Entity']):
        """Define múltiplas entidades para centralizar"""
        self.targets = entities
        self.target = None
        self.mode = "center_on_action"
    
    def update(self, dt: float):
        """Atualiza a posição da câmera"""
        target_x, target_y = self.x, self.y
        
        if self.mode == "follow" and self.target and self.target.is_alive():
            target_x = self.target.x
            target_y = self.target.y
        
        elif self.mode == "center_on_action" and self.targets:
            # Centralizar em todas as entidades vivas
            alive_targets = [e for e in self.targets if e.is_alive()]
            if alive_targets:
                target_x = sum(e.x for e in alive_targets) / len(alive_targets)
                target_y = sum(e.y for e in alive_targets) / len(alive_targets)
        
        elif self.mode == "overview":
            target_x = self.map_width // 2
            target_y = self.map_height // 2
        
        # Suavização
        self.x += (target_x - self.x) * self.smoothing
        self.y += (target_y - self.y) * self.smoothing
        
        # Limitar aos bounds do mapa
        half_screen_w = self.screen_width / (2 * self.zoom)
        half_screen_h = self.screen_height / (2 * self.zoom)
        
        self.x = max(half_screen_w, min(self.map_width - half_screen_w, self.x))
        self.y = max(half_screen_h, min(self.map_height - half_screen_h, self.y))
    
    def get_offset(self) -> Tuple[float, float]:
        """Retorna o offset para desenho"""
        offset_x = self.x - self.screen_width / (2 * self.zoom)
        offset_y = self.y - self.screen_height / (2 * self.zoom)
        return (offset_x, offset_y)
    
    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """Converte coordenadas da tela para mundo"""
        offset_x, offset_y = self.get_offset()
        world_x = screen_x / self.zoom + offset_x
        world_y = screen_y / self.zoom + offset_y
        return (world_x, world_y)
    
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """Converte coordenadas do mundo para tela"""
        offset_x, offset_y = self.get_offset()
        screen_x = (world_x - offset_x) * self.zoom
        screen_y = (world_y - offset_y) * self.zoom
        return (screen_x, screen_y)
    
    def is_visible(self, world_x: float, world_y: float, radius: float = 0) -> bool:
        """Verifica se uma posição está visível na tela"""
        screen_x, screen_y = self.world_to_screen(world_x, world_y)
        margin = radius * self.zoom + 50
        
        return (-margin <= screen_x <= self.screen_width + margin and
                -margin <= screen_y <= self.screen_height + margin)
    
    def zoom_in(self, amount: float = 0.1):
        """Aproxima a câmera"""
        self.zoom = min(self.max_zoom, self.zoom + amount)
    
    def zoom_out(self, amount: float = 0.1):
        """Afasta a câmera"""
        self.zoom = max(self.min_zoom, self.zoom - amount)
    
    def resize(self, screen_width: int, screen_height: int):
        """Atualiza dimensões da tela"""
        self.screen_width = screen_width
        self.screen_height = screen_height
