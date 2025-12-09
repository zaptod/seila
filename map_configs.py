"""
Configuração de Mapas do Jogo
==============================
Define todos os mapas disponíveis, seus tamanhos, temas e obstáculos.
Todos os mapas têm fog of war ativado.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import random
import math


@dataclass
class MapConfig:
    """Configuração completa de um mapa"""
    id: str
    name: str
    description: str
    
    # Tamanho do mapa
    width: int
    height: int
    is_large: bool  # True para mapas grandes (com câmera), False para pequenos
    
    # Cores do tema
    bg_color: Tuple[int, int, int]
    floor_color: Tuple[int, int, int]
    accent_color: Tuple[int, int, int]
    border_color: Tuple[int, int, int]
    obstacle_color: Tuple[int, int, int]
    
    # Configuração de obstáculos
    obstacle_density: float  # 0.0 a 1.0
    obstacle_type: str  # "rocks", "trees", "walls", "ice", "pillars", "mixed"
    
    # Spawn
    spawn_distance: int  # Distância mínima entre spawns
    spawn_clear_radius: int  # Área livre de obstáculos ao redor do spawn
    
    # Efeitos especiais
    has_weather: bool = False
    weather_type: str = "none"  # "rain", "snow", "sandstorm", "fog"
    
    # Fog of war sempre ativo
    fog_enabled: bool = True
    fog_color: Tuple[int, int, int, int] = (20, 20, 30, 180)


# ============================================================================
# MAPAS PEQUENOS (Sem câmera, tela única)
# ============================================================================

SMALL_ARENA = MapConfig(
    id="small_arena",
    name="Arena Pequena",
    description="Arena compacta para combates rápidos e intensos",
    width=1200,
    height=800,
    is_large=False,
    bg_color=(25, 25, 35),
    floor_color=(45, 45, 55),
    accent_color=(80, 60, 50),
    border_color=(100, 80, 70),
    obstacle_color=(70, 70, 80),
    obstacle_density=0.15,
    obstacle_type="pillars",
    spawn_distance=600,
    spawn_clear_radius=100,
)

SMALL_FOREST = MapConfig(
    id="small_forest",
    name="Clareira",
    description="Pequena clareira cercada por árvores densas",
    width=1200,
    height=800,
    is_large=False,
    bg_color=(15, 35, 20),
    floor_color=(40, 60, 35),
    accent_color=(30, 80, 40),
    border_color=(60, 45, 30),
    obstacle_color=(35, 55, 30),
    obstacle_density=0.25,
    obstacle_type="trees",
    spawn_distance=600,
    spawn_clear_radius=100,
)

SMALL_DUNGEON = MapConfig(
    id="small_dungeon",
    name="Sala do Calabouço",
    description="Sala de pedra antiga com colunas destruídas",
    width=1200,
    height=800,
    is_large=False,
    bg_color=(30, 25, 25),
    floor_color=(50, 45, 45),
    accent_color=(70, 50, 40),
    border_color=(80, 60, 50),
    obstacle_color=(60, 55, 50),
    obstacle_density=0.2,
    obstacle_type="walls",
    spawn_distance=600,
    spawn_clear_radius=100,
)

SMALL_ICE = MapConfig(
    id="small_ice",
    name="Lago Congelado",
    description="Superfície de gelo com blocos de neve",
    width=1200,
    height=800,
    is_large=False,
    bg_color=(40, 50, 70),
    floor_color=(180, 200, 220),
    accent_color=(150, 180, 200),
    border_color=(100, 120, 150),
    obstacle_color=(200, 220, 240),
    obstacle_density=0.2,
    obstacle_type="ice",
    spawn_distance=600,
    spawn_clear_radius=100,
    has_weather=True,
    weather_type="snow",
    fog_color=(200, 210, 230, 150),
)

SMALL_DESERT = MapConfig(
    id="small_desert",
    name="Oásis",
    description="Pequeno oásis no meio do deserto",
    width=1200,
    height=800,
    is_large=False,
    bg_color=(80, 60, 30),
    floor_color=(200, 170, 120),
    accent_color=(220, 180, 100),
    border_color=(150, 120, 70),
    obstacle_color=(180, 150, 100),
    obstacle_density=0.15,
    obstacle_type="rocks",
    spawn_distance=600,
    spawn_clear_radius=100,
    has_weather=True,
    weather_type="sandstorm",
    fog_color=(180, 160, 120, 100),
)

# ============================================================================
# MAPAS GRANDES (Com câmera e fog of war extenso)
# ============================================================================

LARGE_ARENA = MapConfig(
    id="large_arena",
    name="Coliseu",
    description="Grande arena de combate com múltiplas áreas",
    width=3000,
    height=3000,
    is_large=True,
    bg_color=(25, 25, 35),
    floor_color=(50, 50, 60),
    accent_color=(90, 70, 60),
    border_color=(120, 100, 90),
    obstacle_color=(80, 80, 90),
    obstacle_density=0.3,
    obstacle_type="pillars",
    spawn_distance=2400,
    spawn_clear_radius=200,
)

LARGE_FOREST = MapConfig(
    id="large_forest",
    name="Floresta Densa",
    description="Floresta extensa com visibilidade reduzida",
    width=4000,
    height=4000,
    is_large=True,
    bg_color=(10, 30, 15),
    floor_color=(35, 55, 30),
    accent_color=(25, 70, 35),
    border_color=(50, 40, 25),
    obstacle_color=(30, 50, 25),
    obstacle_density=0.5,
    obstacle_type="trees",
    spawn_distance=3200,
    spawn_clear_radius=250,
    has_weather=True,
    weather_type="fog",
    fog_color=(30, 50, 40, 200),
)

LARGE_GLACIER = MapConfig(
    id="large_glacier",
    name="Geleira",
    description="Vasta extensão de gelo e neve com nevasca",
    width=3500,
    height=3500,
    is_large=True,
    bg_color=(50, 60, 80),
    floor_color=(200, 215, 230),
    accent_color=(170, 195, 220),
    border_color=(130, 150, 180),
    obstacle_color=(220, 235, 250),
    obstacle_density=0.35,
    obstacle_type="ice",
    spawn_distance=2800,
    spawn_clear_radius=200,
    has_weather=True,
    weather_type="snow",
    fog_color=(220, 230, 245, 180),
)

LARGE_RUINS = MapConfig(
    id="large_ruins",
    name="Ruínas Antigas",
    description="Cidade em ruínas com paredes destruídas",
    width=3500,
    height=3500,
    is_large=True,
    bg_color=(35, 30, 30),
    floor_color=(60, 55, 50),
    accent_color=(80, 60, 50),
    border_color=(100, 80, 70),
    obstacle_color=(70, 65, 60),
    obstacle_density=0.4,
    obstacle_type="walls",
    spawn_distance=2800,
    spawn_clear_radius=200,
)

LARGE_CANYON = MapConfig(
    id="large_canyon",
    name="Cânion Profundo",
    description="Desfiladeiro com passagens estreitas",
    width=4000,
    height=3000,
    is_large=True,
    bg_color=(100, 70, 40),
    floor_color=(180, 140, 100),
    accent_color=(200, 160, 110),
    border_color=(140, 100, 60),
    obstacle_color=(150, 110, 70),
    obstacle_density=0.45,
    obstacle_type="rocks",
    spawn_distance=3000,
    spawn_clear_radius=200,
    has_weather=True,
    weather_type="sandstorm",
    fog_color=(200, 170, 130, 120),
)

LARGE_SWAMP = MapConfig(
    id="large_swamp",
    name="Pântano Sombrio",
    description="Terreno pantanoso com vegetação densa e neblina",
    width=3500,
    height=3500,
    is_large=True,
    bg_color=(20, 35, 25),
    floor_color=(45, 60, 40),
    accent_color=(35, 55, 35),
    border_color=(50, 40, 30),
    obstacle_color=(40, 55, 35),
    obstacle_density=0.45,
    obstacle_type="mixed",
    spawn_distance=2800,
    spawn_clear_radius=200,
    has_weather=True,
    weather_type="fog",
    fog_color=(40, 60, 45, 200),
)

LARGE_VOLCANO = MapConfig(
    id="large_volcano",
    name="Caldeira Vulcânica",
    description="Interior de vulcão com lava e rochas ardentes",
    width=3000,
    height=3000,
    is_large=True,
    bg_color=(50, 20, 10),
    floor_color=(80, 40, 30),
    accent_color=(200, 80, 30),
    border_color=(120, 50, 20),
    obstacle_color=(90, 50, 35),
    obstacle_density=0.35,
    obstacle_type="rocks",
    spawn_distance=2400,
    spawn_clear_radius=200,
    fog_color=(100, 40, 20, 150),
)

LARGE_CASTLE = MapConfig(
    id="large_castle",
    name="Castelo em Ruínas",
    description="Fortaleza abandonada com salões e corredores",
    width=4000,
    height=4000,
    is_large=True,
    bg_color=(30, 30, 40),
    floor_color=(55, 55, 65),
    accent_color=(70, 60, 80),
    border_color=(90, 80, 100),
    obstacle_color=(65, 65, 75),
    obstacle_density=0.5,
    obstacle_type="walls",
    spawn_distance=3200,
    spawn_clear_radius=250,
)

# ============================================================================
# REGISTRO DE MAPAS
# ============================================================================

ALL_MAPS: Dict[str, MapConfig] = {
    # Pequenos
    "small_arena": SMALL_ARENA,
    "small_forest": SMALL_FOREST,
    "small_dungeon": SMALL_DUNGEON,
    "small_ice": SMALL_ICE,
    "small_desert": SMALL_DESERT,
    # Grandes
    "large_arena": LARGE_ARENA,
    "large_forest": LARGE_FOREST,
    "large_glacier": LARGE_GLACIER,
    "large_ruins": LARGE_RUINS,
    "large_canyon": LARGE_CANYON,
    "large_swamp": LARGE_SWAMP,
    "large_volcano": LARGE_VOLCANO,
    "large_castle": LARGE_CASTLE,
}

SMALL_MAPS = [m for m in ALL_MAPS.values() if not m.is_large]
LARGE_MAPS = [m for m in ALL_MAPS.values() if m.is_large]


def get_map(map_id: str) -> Optional[MapConfig]:
    """Retorna configuração de um mapa pelo ID"""
    return ALL_MAPS.get(map_id)


def list_maps(large_only: bool = False, small_only: bool = False) -> List[MapConfig]:
    """Lista mapas disponíveis"""
    if large_only:
        return LARGE_MAPS.copy()
    elif small_only:
        return SMALL_MAPS.copy()
    return list(ALL_MAPS.values())


def get_spawn_positions(map_config: MapConfig, num_teams: int = 2, 
                        team_size: int = 1) -> List[List[Tuple[float, float]]]:
    """
    Calcula posições de spawn para os times.
    Retorna lista de listas: [[posições time 1], [posições time 2]]
    """
    positions = [[], []]
    
    # Centro do mapa
    cx, cy = map_config.width / 2, map_config.height / 2
    
    # Distância do centro para cada time
    spawn_dist = map_config.spawn_distance / 2
    
    # Ângulos para os times (opostos)
    angle1 = math.pi  # Esquerda
    angle2 = 0  # Direita
    
    # Centro de spawn de cada time
    team1_cx = cx + math.cos(angle1) * spawn_dist
    team1_cy = cy + math.sin(angle1) * spawn_dist
    team2_cx = cx + math.cos(angle2) * spawn_dist
    team2_cy = cy + math.sin(angle2) * spawn_dist
    
    # Espaçamento entre membros do mesmo time
    spacing = 60
    
    # Calcular posições para cada membro do time
    for i in range(team_size):
        # Time 1 - distribuir verticalmente
        offset = (i - (team_size - 1) / 2) * spacing
        positions[0].append((team1_cx, team1_cy + offset))
        
        # Time 2 - distribuir verticalmente
        positions[1].append((team2_cx, team2_cy + offset))
    
    return positions
