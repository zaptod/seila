"""
Sistema de Mapas
================
Mapas variados para o jogo com diferentes temas visuais.
"""

import pygame
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class MapTheme:
    """Definição de um tema de mapa"""
    name: str
    display_name: str
    bg_color: Tuple[int, int, int]
    floor_color: Tuple[int, int, int]
    accent_color: Tuple[int, int, int]
    detail_color: Tuple[int, int, int]
    border_color: Tuple[int, int, int]
    description: str
    # Elementos decorativos
    draw_func: str = "default"


class MapRegistry:
    """Registro de todos os mapas disponíveis"""
    _maps: Dict[str, MapTheme] = {}
    
    @classmethod
    def register(cls, map_id: str, theme: MapTheme):
        cls._maps[map_id] = theme
    
    @classmethod
    def get(cls, map_id: str) -> MapTheme:
        return cls._maps.get(map_id)
    
    @classmethod
    def get_all(cls) -> Dict[str, MapTheme]:
        return cls._maps.copy()
    
    @classmethod
    def list_maps(cls) -> List[str]:
        return list(cls._maps.keys())


# ============================================================================
# DEFINIÇÃO DOS MAPAS
# ============================================================================

# 1. Arena Clássica
MapRegistry.register("arena", MapTheme(
    name="arena",
    display_name="Arena Clássica",
    bg_color=(20, 20, 30),
    floor_color=(40, 40, 50),
    accent_color=(80, 60, 40),
    detail_color=(60, 60, 70),
    border_color=(100, 80, 60),
    description="A arena tradicional de combate"
))

# 2. Floresta Sombria
MapRegistry.register("forest", MapTheme(
    name="forest",
    display_name="Floresta Sombria",
    bg_color=(15, 30, 20),
    floor_color=(35, 55, 35),
    accent_color=(25, 80, 35),
    detail_color=(60, 90, 50),
    border_color=(80, 60, 40),
    description="Uma clareira na floresta densa"
))

# 3. Deserto Escaldante
MapRegistry.register("desert", MapTheme(
    name="desert",
    display_name="Deserto Escaldante",
    bg_color=(60, 45, 25),
    floor_color=(200, 170, 120),
    accent_color=(220, 180, 100),
    detail_color=(180, 140, 80),
    border_color=(140, 100, 60),
    description="Areias quentes sob sol implacável"
))

# 4. Caverna de Cristal
MapRegistry.register("cave", MapTheme(
    name="cave",
    display_name="Caverna de Cristal",
    bg_color=(20, 15, 35),
    floor_color=(40, 35, 60),
    accent_color=(100, 80, 180),
    detail_color=(80, 60, 140),
    border_color=(120, 100, 200),
    description="Cavernas iluminadas por cristais"
))

# 5. Castelo Medieval
MapRegistry.register("castle", MapTheme(
    name="castle",
    display_name="Castelo Medieval",
    bg_color=(25, 25, 30),
    floor_color=(60, 55, 50),
    accent_color=(120, 100, 80),
    detail_color=(80, 70, 60),
    border_color=(150, 130, 100),
    description="O pátio de um castelo antigo"
))

# 6. Vulcão Ardente
MapRegistry.register("volcano", MapTheme(
    name="volcano",
    display_name="Vulcão Ardente",
    bg_color=(40, 15, 10),
    floor_color=(60, 30, 20),
    accent_color=(200, 80, 30),
    detail_color=(255, 120, 40),
    border_color=(180, 60, 20),
    description="Plataforma sobre lava incandescente"
))

# 7. Praia Tropical
MapRegistry.register("beach", MapTheme(
    name="beach",
    display_name="Praia Tropical",
    bg_color=(30, 100, 150),
    floor_color=(220, 200, 150),
    accent_color=(100, 180, 220),
    detail_color=(240, 220, 180),
    border_color=(150, 130, 100),
    description="Areias douradas à beira-mar"
))

# 8. Cidade Noturna
MapRegistry.register("city", MapTheme(
    name="city",
    display_name="Cidade Noturna",
    bg_color=(15, 15, 25),
    floor_color=(45, 45, 55),
    accent_color=(255, 200, 50),
    detail_color=(80, 80, 100),
    border_color=(70, 70, 90),
    description="Telhado de prédio na metrópole"
))

# 9. Neve Eterna
MapRegistry.register("snow", MapTheme(
    name="snow",
    display_name="Neve Eterna",
    bg_color=(40, 50, 70),
    floor_color=(220, 230, 240),
    accent_color=(180, 200, 220),
    detail_color=(200, 210, 230),
    border_color=(150, 170, 200),
    description="Planícies congeladas do norte"
))

# 10. Espaço Sideral
MapRegistry.register("space", MapTheme(
    name="space",
    display_name="Espaço Sideral",
    bg_color=(5, 5, 15),
    floor_color=(25, 25, 40),
    accent_color=(100, 50, 200),
    detail_color=(50, 30, 100),
    border_color=(80, 40, 150),
    description="Plataforma no vazio cósmico"
))


class MapRenderer:
    """Renderiza os mapas com seus elementos decorativos"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.margin = 50
        # Cache de elementos decorativos por mapa
        self._decoration_cache = {}
        self._stars_cache = None
        self._tree_positions = None
        self._crystal_positions = None
    
    def resize(self, width: int, height: int):
        """Atualiza dimensões e limpa cache"""
        self.width = width
        self.height = height
        self._decoration_cache.clear()
        self._stars_cache = None
        self._tree_positions = None
        self._crystal_positions = None
    
    def _generate_stars(self, count: int = 100) -> List[Tuple[int, int, int]]:
        """Gera posições de estrelas para mapa espacial"""
        if self._stars_cache is None:
            random.seed(42)  # Seed fixa para consistência
            self._stars_cache = [
                (random.randint(0, self.width), 
                 random.randint(0, self.height),
                 random.randint(1, 3))  # tamanho
                for _ in range(count)
            ]
        return self._stars_cache
    
    def _generate_trees(self, count: int = 20) -> List[Tuple[int, int]]:
        """Gera posições de árvores para floresta"""
        if self._tree_positions is None:
            random.seed(43)
            positions = []
            for _ in range(count):
                # Árvores nas bordas
                side = random.choice(['top', 'bottom', 'left', 'right'])
                if side == 'top':
                    x = random.randint(self.margin, self.width - self.margin)
                    y = random.randint(10, self.margin - 10)
                elif side == 'bottom':
                    x = random.randint(self.margin, self.width - self.margin)
                    y = random.randint(self.height - self.margin + 10, self.height - 10)
                elif side == 'left':
                    x = random.randint(10, self.margin - 10)
                    y = random.randint(self.margin, self.height - self.margin)
                else:
                    x = random.randint(self.width - self.margin + 10, self.width - 10)
                    y = random.randint(self.margin, self.height - self.margin)
                positions.append((x, y))
            self._tree_positions = positions
        return self._tree_positions
    
    def _generate_crystals(self, count: int = 15) -> List[Tuple[int, int, int, Tuple[int, int, int]]]:
        """Gera cristais para caverna"""
        if self._crystal_positions is None:
            random.seed(44)
            colors = [
                (150, 100, 255), (100, 200, 255), (255, 100, 200),
                (100, 255, 150), (255, 200, 100)
            ]
            crystals = []
            for _ in range(count):
                x = random.randint(self.margin - 30, self.width - self.margin + 30)
                y = random.choice([
                    random.randint(5, self.margin - 20),
                    random.randint(self.height - self.margin + 20, self.height - 5)
                ])
                size = random.randint(15, 35)
                color = random.choice(colors)
                crystals.append((x, y, size, color))
            self._crystal_positions = crystals
        return self._crystal_positions
    
    def draw(self, screen: pygame.Surface, map_id: str, time_offset: float = 0):
        """Desenha o mapa completo"""
        theme = MapRegistry.get(map_id)
        if not theme:
            theme = MapRegistry.get("arena")
        
        # Fundo
        screen.fill(theme.bg_color)
        
        # Desenhar elementos específicos do mapa
        draw_method = getattr(self, f'_draw_{map_id}', self._draw_default)
        draw_method(screen, theme, time_offset)
        
        # Arena (área de jogo)
        arena_rect = pygame.Rect(
            self.margin, self.margin,
            self.width - 2 * self.margin,
            self.height - 2 * self.margin
        )
        pygame.draw.rect(screen, theme.floor_color, arena_rect)
        pygame.draw.rect(screen, theme.border_color, arena_rect, 4)
        
        # Linhas decorativas no centro
        center_x = self.width // 2
        center_y = self.height // 2
        
        # Círculo central
        pygame.draw.circle(screen, theme.detail_color, (center_x, center_y), 80, 2)
        pygame.draw.circle(screen, theme.detail_color, (center_x, center_y), 5)
        
        # Linha do meio
        pygame.draw.line(screen, theme.detail_color, 
                        (center_x, self.margin), 
                        (center_x, self.height - self.margin), 2)
    
    def _draw_default(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Desenho padrão (arena clássica)"""
        # Padrão quadriculado sutil
        for x in range(self.margin, self.width - self.margin, 50):
            pygame.draw.line(screen, theme.detail_color, 
                           (x, self.margin), (x, self.height - self.margin), 1)
        for y in range(self.margin, self.height - self.margin, 50):
            pygame.draw.line(screen, theme.detail_color,
                           (self.margin, y), (self.width - self.margin, y), 1)
    
    def _draw_arena(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Arena clássica"""
        self._draw_default(screen, theme, time_offset)
    
    def _draw_forest(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Floresta com árvores"""
        # Árvores nas bordas
        trees = self._generate_trees(25)
        for tx, ty in trees:
            # Tronco
            pygame.draw.rect(screen, (60, 40, 20), 
                           (tx - 5, ty - 10, 10, 30))
            # Copa
            pygame.draw.circle(screen, theme.accent_color, (tx, ty - 20), 20)
            pygame.draw.circle(screen, (30, 100, 40), (tx - 10, ty - 15), 15)
            pygame.draw.circle(screen, (40, 90, 35), (tx + 10, ty - 18), 12)
        
        # Grama no chão
        random.seed(45)
        for _ in range(50):
            gx = random.randint(self.margin + 10, self.width - self.margin - 10)
            gy = random.randint(self.margin + 10, self.height - self.margin - 10)
            pygame.draw.line(screen, theme.detail_color,
                           (gx, gy), (gx + random.randint(-5, 5), gy - random.randint(5, 12)), 1)
    
    def _draw_desert(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Deserto com dunas"""
        # Dunas de areia
        random.seed(46)
        for i in range(8):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            w = random.randint(100, 300)
            h = random.randint(30, 80)
            color = (
                theme.floor_color[0] + random.randint(-20, 20),
                theme.floor_color[1] + random.randint(-20, 20),
                theme.floor_color[2] + random.randint(-20, 20)
            )
            pygame.draw.ellipse(screen, color, (x - w//2, y - h//2, w, h))
        
        # Cactos nas bordas
        for i in range(6):
            cx = random.randint(10, self.margin - 20) if i % 2 == 0 else random.randint(self.width - self.margin + 20, self.width - 10)
            cy = random.randint(self.margin, self.height - self.margin)
            # Tronco
            pygame.draw.rect(screen, (40, 80, 40), (cx - 5, cy - 30, 10, 50))
            # Braços
            pygame.draw.rect(screen, (40, 80, 40), (cx - 20, cy - 20, 15, 8))
            pygame.draw.rect(screen, (40, 80, 40), (cx - 20, cy - 30, 8, 15))
            pygame.draw.rect(screen, (40, 80, 40), (cx + 5, cy - 10, 15, 8))
            pygame.draw.rect(screen, (40, 80, 40), (cx + 12, cy - 25, 8, 20))
    
    def _draw_cave(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Caverna com cristais brilhantes"""
        # Cristais nas bordas
        crystals = self._generate_crystals(18)
        for cx, cy, size, color in crystals:
            # Cristal (triângulo)
            glow = (
                min(255, color[0] + 50),
                min(255, color[1] + 50),
                min(255, color[2] + 50)
            )
            # Brilho pulsante
            pulse = math.sin(time_offset * 2 + cx * 0.1) * 0.3 + 0.7
            actual_color = tuple(int(c * pulse) for c in color)
            
            points = [
                (cx, cy - size),
                (cx - size // 2, cy + size // 3),
                (cx + size // 2, cy + size // 3)
            ]
            pygame.draw.polygon(screen, actual_color, points)
            pygame.draw.polygon(screen, glow, points, 2)
        
        # Estalactites no topo
        random.seed(47)
        for i in range(15):
            sx = random.randint(self.margin, self.width - self.margin)
            sh = random.randint(20, 50)
            points = [
                (sx - 8, 0),
                (sx + 8, 0),
                (sx, sh)
            ]
            pygame.draw.polygon(screen, theme.detail_color, points)
    
    def _draw_castle(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Castelo medieval"""
        # Torres nas laterais
        tower_positions = [
            (20, 20), (self.width - 60, 20),
            (20, self.height - 80), (self.width - 60, self.height - 80)
        ]
        for tx, ty in tower_positions:
            # Base da torre
            pygame.draw.rect(screen, theme.detail_color, (tx, ty, 40, 60))
            # Ameias
            for i in range(4):
                pygame.draw.rect(screen, theme.detail_color, (tx + i * 10, ty - 10, 8, 12))
            # Janela
            pygame.draw.rect(screen, (20, 20, 30), (tx + 15, ty + 25, 10, 15))
        
        # Bandeiras
        random.seed(48)
        flag_colors = [(200, 50, 50), (50, 50, 200), (200, 200, 50)]
        for i, (tx, ty) in enumerate(tower_positions[:2]):
            # Mastro
            pygame.draw.line(screen, (100, 80, 60), 
                           (tx + 20, ty - 10), (tx + 20, ty - 50), 3)
            # Bandeira
            wave = math.sin(time_offset * 3 + i) * 5
            flag_points = [
                (tx + 22, ty - 48),
                (tx + 50 + wave, ty - 40),
                (tx + 22, ty - 30)
            ]
            pygame.draw.polygon(screen, flag_colors[i % len(flag_colors)], flag_points)
    
    def _draw_volcano(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Vulcão com lava"""
        # Lava ao redor (animada)
        random.seed(49)
        for i in range(30):
            lx = random.randint(0, self.width)
            ly = random.randint(0, self.height)
            
            # Pular área da arena
            if self.margin < lx < self.width - self.margin and self.margin < ly < self.height - self.margin:
                continue
            
            pulse = math.sin(time_offset * 4 + i * 0.5) * 0.3 + 0.7
            lava_color = (
                int(255 * pulse),
                int(100 * pulse),
                int(30 * pulse)
            )
            size = random.randint(15, 40)
            pygame.draw.circle(screen, lava_color, (lx, ly), size)
        
        # Rochas
        for i in range(10):
            rx = random.randint(5, self.margin - 10) if i % 2 == 0 else random.randint(self.width - self.margin + 10, self.width - 5)
            ry = random.randint(self.margin, self.height - self.margin)
            rw = random.randint(20, 40)
            rh = random.randint(15, 30)
            pygame.draw.ellipse(screen, (50, 35, 30), (rx - rw//2, ry - rh//2, rw, rh))
    
    def _draw_beach(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Praia tropical"""
        # Ondas do mar (lado esquerdo ou direito)
        wave_x = self.margin - 30
        for i in range(10):
            wave_offset = math.sin(time_offset * 2 + i * 0.8) * 10
            y = self.margin + i * ((self.height - 2 * self.margin) // 10)
            pygame.draw.arc(screen, theme.accent_color,
                          (wave_x + wave_offset - 20, y - 10, 40, 20),
                          0, math.pi, 3)
        
        # Conchas na areia
        random.seed(50)
        for _ in range(8):
            sx = random.randint(self.margin + 20, self.width - self.margin - 20)
            sy = random.randint(self.margin + 20, self.height - self.margin - 20)
            pygame.draw.arc(screen, (240, 200, 180),
                          (sx - 8, sy - 5, 16, 10), 0, math.pi, 2)
        
        # Palmeiras nas bordas
        palm_positions = [(self.width - 30, 80), (self.width - 35, self.height - 100)]
        for px, py in palm_positions:
            # Tronco curvo
            pygame.draw.arc(screen, (100, 70, 40),
                          (px - 30, py - 60, 60, 120), -0.5, 1.5, 8)
            # Folhas
            for angle in range(-60, 61, 30):
                rad = math.radians(angle - 90)
                leaf_len = 40
                ex = px + 5 + math.cos(rad) * leaf_len
                ey = py - 60 + math.sin(rad) * leaf_len
                pygame.draw.line(screen, (40, 120, 40), (px + 5, py - 60), (ex, ey), 4)
    
    def _draw_city(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Cidade noturna"""
        # Prédios ao fundo
        random.seed(51)
        for i in range(20):
            bx = random.randint(0, self.width)
            bw = random.randint(30, 80)
            bh = random.randint(100, 300)
            by = self.height - bh
            
            # Prédio
            pygame.draw.rect(screen, (30, 30, 45), (bx, by, bw, bh))
            
            # Janelas (algumas acesas)
            for wy in range(by + 10, self.height - 10, 20):
                for wx in range(bx + 5, bx + bw - 10, 15):
                    is_lit = random.random() > 0.5
                    window_color = theme.accent_color if is_lit else (40, 40, 50)
                    pygame.draw.rect(screen, window_color, (wx, wy, 8, 12))
        
        # Letreiros neon piscando
        neon_colors = [(255, 50, 100), (50, 255, 100), (100, 50, 255)]
        for i, color in enumerate(neon_colors):
            nx = 100 + i * 300
            if nx < self.width - 50:
                pulse = (math.sin(time_offset * 5 + i * 2) + 1) / 2
                if pulse > 0.3:
                    pygame.draw.rect(screen, color, (nx, 30, 60, 20), 2)
    
    def _draw_snow(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Neve eterna"""
        # Neve caindo
        random.seed(52)
        for i in range(80):
            # Posição animada
            sx = (random.randint(0, self.width) + int(time_offset * 20)) % self.width
            sy = (random.randint(0, self.height) + int(time_offset * 50 + i * 10)) % self.height
            size = random.randint(2, 5)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), size)
        
        # Montes de neve nas bordas
        for i in range(6):
            mx = random.randint(0, self.width)
            my = random.choice([random.randint(5, self.margin - 10),
                              random.randint(self.height - self.margin + 10, self.height - 5)])
            mw = random.randint(60, 150)
            mh = random.randint(20, 40)
            pygame.draw.ellipse(screen, (240, 245, 255), (mx - mw//2, my - mh//2, mw, mh))
    
    def _draw_space(self, screen: pygame.Surface, theme: MapTheme, time_offset: float):
        """Espaço sideral"""
        # Estrelas
        stars = self._generate_stars(150)
        for sx, sy, size in stars:
            # Estrelas piscando
            twinkle = (math.sin(time_offset * 3 + sx * 0.01 + sy * 0.01) + 1) / 2
            brightness = int(150 + 105 * twinkle)
            pygame.draw.circle(screen, (brightness, brightness, brightness), (sx, sy), size)
        
        # Nebulosa
        random.seed(53)
        for i in range(5):
            nx = random.randint(0, self.width)
            ny = random.randint(0, self.height)
            nr = random.randint(50, 150)
            
            # Cor da nebulosa
            nebula_colors = [
                (100, 50, 150, 30),
                (50, 100, 150, 30),
                (150, 50, 100, 30)
            ]
            color = nebula_colors[i % len(nebula_colors)]
            
            # Desenhar como círculos semi-transparentes sobrepostos
            for r in range(nr, 0, -10):
                alpha = max(0, min(255, color[3] * (r / nr)))
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(surf, (*color[:3], int(alpha)), (r, r), r)
                screen.blit(surf, (nx - r, ny - r))
        
        # Planeta distante
        planet_x = self.width - 100
        planet_y = 80
        pygame.draw.circle(screen, (150, 100, 80), (planet_x, planet_y), 40)
        pygame.draw.circle(screen, (120, 80, 60), (planet_x - 10, planet_y - 10), 35)
        # Anel
        pygame.draw.ellipse(screen, (180, 150, 120), 
                          (planet_x - 60, planet_y - 10, 120, 20), 2)
