"""
Sistema de Armas
================
Sistema modular e extensível para armas com registro automático.
Fácil adicionar novas armas apenas criando uma nova classe.
Inclui armas de combate, suporte, controle e longa distância.
"""

import pygame
import math
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass
from stats import WeaponStats, StatusEffect, StatusEffectType

if TYPE_CHECKING:
    from entities import Entity


# ============================================================================
# REGISTRO DE ARMAS
# ============================================================================

class WeaponRegistry:
    """Registro global de todas as armas disponíveis"""
    _weapons: Dict[str, Type['Weapon']] = {}
    
    @classmethod
    def register(cls, weapon_id: str):
        """Decorator para registrar uma arma"""
        def decorator(weapon_class: Type['Weapon']):
            cls._weapons[weapon_id] = weapon_class
            weapon_class.weapon_id = weapon_id
            return weapon_class
        return decorator
    
    @classmethod
    def create(cls, weapon_id: str, owner: 'Entity') -> 'Weapon':
        """Cria uma instância de arma pelo ID"""
        if weapon_id not in cls._weapons:
            raise ValueError(f"Arma não encontrada: {weapon_id}")
        return cls._weapons[weapon_id](owner)
    
    @classmethod
    def get_all(cls) -> Dict[str, Type['Weapon']]:
        """Retorna todas as armas registradas"""
        return cls._weapons.copy()
    
    @classmethod
    def list_weapons(cls) -> List[str]:
        """Lista IDs de todas as armas"""
        return list(cls._weapons.keys())


# ============================================================================
# CLASSE BASE DE ARMA
# ============================================================================

class Weapon(ABC):
    """Classe base abstrata para todas as armas"""
    
    weapon_id: str = "base"
    display_name: str = "Arma Base"
    description: str = "Uma arma básica"
    
    def __init__(self, owner: 'Entity'):
        self.owner = owner
        self.stats = self.get_default_stats()
        
        # Estado
        self.is_attacking = False
        self.attack_cooldown = 0
        self.attack_timer = 0
        
        # Para IA - ações disponíveis
        self.can_attack = True
    
    @abstractmethod
    def get_default_stats(self) -> WeaponStats:
        """Retorna os stats padrão da arma"""
        pass
    
    @abstractmethod
    def get_hitbox(self) -> Optional[Dict]:
        """Retorna informações da hitbox para detecção de colisão"""
        pass
    
    @abstractmethod
    def draw(self, screen: pygame.Surface):
        """Desenha a arma"""
        pass
    
    def attack(self) -> bool:
        """
        Inicia um ataque.
        Retorna True se o ataque foi iniciado, False se ainda em cooldown.
        """
        if self.attack_cooldown <= 0 and not self.is_attacking:
            self.is_attacking = True
            effective_cooldown = self.stats.attack_cooldown / self.owner.stats_manager.get_stats().attack_speed
            self.attack_cooldown = effective_cooldown
            self.attack_timer = self.stats.attack_duration
            self.on_attack_start()
            return True
        return False
    
    def on_attack_start(self):
        """Callback quando um ataque inicia (para override)"""
        pass
    
    def on_attack_end(self):
        """Callback quando um ataque termina (para override)"""
        pass
    
    def on_hit(self, target: 'Entity', damage: float):
        """Callback quando a arma acerta um alvo (para override)"""
        pass
    
    def update(self, dt: float):
        """Atualiza o estado da arma"""
        # Atualizar cooldown
        if self.attack_cooldown > 0:
            self.attack_cooldown -= dt
        
        # Atualizar ataque
        if self.is_attacking:
            self.attack_timer -= dt
            self.update_attack(dt)
            
            if self.attack_timer <= 0:
                self.is_attacking = False
                self.on_attack_end()
        
        # Atualizar flag para IA
        self.can_attack = self.attack_cooldown <= 0 and not self.is_attacking
    
    def update_attack(self, dt: float):
        """Atualiza a lógica do ataque (para override)"""
        pass
    
    def calculate_damage(self) -> float:
        """Calcula o dano do ataque atual"""
        return self.owner.stats_manager.calculate_damage(
            self.stats.base_damage, 
            self.stats
        )
    
    def get_state(self) -> Dict:
        """Retorna estado da arma para IA/serialização"""
        return {
            'weapon_id': self.weapon_id,
            'is_attacking': self.is_attacking,
            'attack_cooldown': self.attack_cooldown,
            'can_attack': self.can_attack,
            'cooldown_progress': 1 - (self.attack_cooldown / self.stats.attack_cooldown) if self.stats.attack_cooldown > 0 else 1
        }


# ============================================================================
# ARMAS IMPLEMENTADAS
# ============================================================================

@WeaponRegistry.register("sword")
class Sword(Weapon):
    """Espada - arma corpo a corpo balanceada"""
    
    display_name = "Espada"
    description = "Arma corpo a corpo equilibrada com bom alcance e dano."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        
        # Animação específica da espada
        self.swing_angle = 0
        self.swing_range = math.pi * 0.6  # Swing reduzido
        self.current_angle = 0
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=14,
            attack_cooldown=0.75,
            attack_duration=0.55,
            range=50,
            width=8,
            knockback_force=14,
            critical_chance=0.12,
            critical_multiplier=1.8
        )
    
    def on_attack_start(self):
        self.swing_angle = -self.swing_range / 2
    
    def update_attack(self, dt: float):
        progress = 1 - (self.attack_timer / self.stats.attack_duration)
        self.swing_angle = -self.swing_range / 2 + self.swing_range * progress
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle + self.swing_angle
    
    def get_sword_points(self) -> Tuple[List[Tuple[float, float]], Tuple[float, float], Tuple[float, float]]:
        """Calcula os pontos da espada"""
        effective_range = self.stats.range * self.owner.stats_manager.get_stats().attack_range
        
        base_distance = self.owner.radius + 5
        base_x = self.owner.x + math.cos(self.current_angle) * base_distance
        base_y = self.owner.y + math.sin(self.current_angle) * base_distance
        
        tip_x = base_x + math.cos(self.current_angle) * effective_range
        tip_y = base_y + math.sin(self.current_angle) * effective_range
        
        perpendicular = self.current_angle + math.pi / 2
        half_width = self.stats.width / 2
        px = math.cos(perpendicular) * half_width
        py = math.sin(perpendicular) * half_width
        
        points = [
            (base_x + px, base_y + py),
            (base_x - px, base_y - py),
            (tip_x - px, tip_y - py),
            (tip_x + px, tip_y + py)
        ]
        
        return points, (base_x, base_y), (tip_x, tip_y)
    
    def get_hitbox(self) -> Optional[Dict]:
        points, base, tip = self.get_sword_points()
        return {
            'type': 'line',
            'start': base,
            'end': tip,
            'points': points,
            'active': self.is_attacking,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': self.stats.armor_penetration
        }
    
    def draw(self, screen: pygame.Surface):
        points, base, tip = self.get_sword_points()
        
        if self.is_attacking:
            blade_color = (255, 255, 200)
            edge_color = (255, 200, 100)
        else:
            blade_color = (200, 200, 200)
            edge_color = (150, 150, 150)
        
        pygame.draw.polygon(screen, blade_color, points)
        pygame.draw.polygon(screen, edge_color, points, 2)
        
        # Guarda
        guard_size = 12
        perpendicular = self.current_angle + math.pi / 2
        guard_x1 = base[0] + math.cos(perpendicular) * guard_size
        guard_y1 = base[1] + math.sin(perpendicular) * guard_size
        guard_x2 = base[0] - math.cos(perpendicular) * guard_size
        guard_y2 = base[1] - math.sin(perpendicular) * guard_size
        pygame.draw.line(screen, (139, 69, 19), 
                        (int(guard_x1), int(guard_y1)), 
                        (int(guard_x2), int(guard_y2)), 4)
        
        # Cabo
        handle_length = 15
        handle_x = base[0] - math.cos(self.current_angle) * handle_length
        handle_y = base[1] - math.sin(self.current_angle) * handle_length
        pygame.draw.line(screen, (101, 67, 33), 
                        (int(base[0]), int(base[1])), 
                        (int(handle_x), int(handle_y)), 6)


@WeaponRegistry.register("greatsword")
class Greatsword(Weapon):
    """Espadão - arma lenta mas com alto dano e alcance"""
    
    display_name = "Espadão"
    description = "Arma pesada e lenta, mas com dano devastador e grande alcance."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.swing_angle = 0
        self.swing_range = math.pi * 0.8  # Swing reduzido
        self.current_angle = 0
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=28,
            attack_cooldown=1.8,
            attack_duration=1.1,
            range=70,
            width=12,
            knockback_force=28,
            critical_chance=0.12,
            critical_multiplier=2.2,
            armor_penetration=0.15
        )
    
    def on_attack_start(self):
        self.swing_angle = -self.swing_range / 2
    
    def update_attack(self, dt: float):
        progress = 1 - (self.attack_timer / self.stats.attack_duration)
        self.swing_angle = -self.swing_range / 2 + self.swing_range * progress
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle + self.swing_angle
    
    def get_sword_points(self):
        effective_range = self.stats.range * self.owner.stats_manager.get_stats().attack_range
        
        base_distance = self.owner.radius + 5
        base_x = self.owner.x + math.cos(self.current_angle) * base_distance
        base_y = self.owner.y + math.sin(self.current_angle) * base_distance
        
        tip_x = base_x + math.cos(self.current_angle) * effective_range
        tip_y = base_y + math.sin(self.current_angle) * effective_range
        
        perpendicular = self.current_angle + math.pi / 2
        half_width = self.stats.width / 2
        px = math.cos(perpendicular) * half_width
        py = math.sin(perpendicular) * half_width
        
        points = [
            (base_x + px, base_y + py),
            (base_x - px, base_y - py),
            (tip_x - px, tip_y - py),
            (tip_x + px, tip_y + py)
        ]
        
        return points, (base_x, base_y), (tip_x, tip_y)
    
    def get_hitbox(self) -> Optional[Dict]:
        points, base, tip = self.get_sword_points()
        return {
            'type': 'line',
            'start': base,
            'end': tip,
            'points': points,
            'active': self.is_attacking,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': self.stats.armor_penetration
        }
    
    def draw(self, screen: pygame.Surface):
        points, base, tip = self.get_sword_points()
        
        if self.is_attacking:
            blade_color = (255, 220, 180)
            edge_color = (255, 180, 100)
        else:
            blade_color = (180, 180, 190)
            edge_color = (140, 140, 150)
        
        pygame.draw.polygon(screen, blade_color, points)
        pygame.draw.polygon(screen, edge_color, points, 3)
        
        # Guarda grande
        guard_size = 18
        perpendicular = self.current_angle + math.pi / 2
        guard_x1 = base[0] + math.cos(perpendicular) * guard_size
        guard_y1 = base[1] + math.sin(perpendicular) * guard_size
        guard_x2 = base[0] - math.cos(perpendicular) * guard_size
        guard_y2 = base[1] - math.sin(perpendicular) * guard_size
        pygame.draw.line(screen, (100, 50, 20), 
                        (int(guard_x1), int(guard_y1)), 
                        (int(guard_x2), int(guard_y2)), 6)
        
        # Cabo longo
        handle_length = 25
        handle_x = base[0] - math.cos(self.current_angle) * handle_length
        handle_y = base[1] - math.sin(self.current_angle) * handle_length
        pygame.draw.line(screen, (80, 50, 30), 
                        (int(base[0]), int(base[1])), 
                        (int(handle_x), int(handle_y)), 8)


@WeaponRegistry.register("dagger")
class Dagger(Weapon):
    """Adaga - arma rápida com baixo dano mas alta crítica"""
    
    display_name = "Adaga"
    description = "Arma leve e rápida com alta chance de crítico."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.swing_angle = 0
        self.swing_range = math.pi * 0.3  # Swing reduzido
        self.current_angle = 0
        self.stab_offset = 0  # Para animação de estocada
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=7,
            attack_cooldown=0.35,
            attack_duration=0.25,
            range=30,
            width=4,
            knockback_force=4,
            critical_chance=0.25,
            critical_multiplier=2.5,
            lifesteal=0.08
        )
    
    def on_attack_start(self):
        self.stab_offset = 0
    
    def update_attack(self, dt: float):
        progress = 1 - (self.attack_timer / self.stats.attack_duration)
        # Animação de estocada (vai e volta)
        if progress < 0.5:
            self.stab_offset = progress * 2 * 20  # Avança
        else:
            self.stab_offset = (1 - progress) * 2 * 20  # Recua
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle
    
    def get_dagger_points(self):
        effective_range = self.stats.range * self.owner.stats_manager.get_stats().attack_range
        
        base_distance = self.owner.radius + 5 + self.stab_offset
        base_x = self.owner.x + math.cos(self.current_angle) * base_distance
        base_y = self.owner.y + math.sin(self.current_angle) * base_distance
        
        tip_x = base_x + math.cos(self.current_angle) * effective_range
        tip_y = base_y + math.sin(self.current_angle) * effective_range
        
        perpendicular = self.current_angle + math.pi / 2
        half_width = self.stats.width / 2
        px = math.cos(perpendicular) * half_width
        py = math.sin(perpendicular) * half_width
        
        # Forma de adaga (mais larga na base, pontuda)
        mid_x = base_x + math.cos(self.current_angle) * effective_range * 0.3
        mid_y = base_y + math.sin(self.current_angle) * effective_range * 0.3
        
        points = [
            (base_x + px * 0.5, base_y + py * 0.5),
            (mid_x + px * 1.5, mid_y + py * 1.5),
            (tip_x, tip_y),
            (mid_x - px * 1.5, mid_y - py * 1.5),
            (base_x - px * 0.5, base_y - py * 0.5),
        ]
        
        return points, (base_x, base_y), (tip_x, tip_y)
    
    def get_hitbox(self) -> Optional[Dict]:
        points, base, tip = self.get_dagger_points()
        return {
            'type': 'line',
            'start': base,
            'end': tip,
            'points': points,
            'active': self.is_attacking,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': self.stats.armor_penetration
        }
    
    def draw(self, screen: pygame.Surface):
        points, base, tip = self.get_dagger_points()
        
        if self.is_attacking:
            blade_color = (220, 220, 255)
            edge_color = (180, 180, 255)
        else:
            blade_color = (190, 190, 200)
            edge_color = (150, 150, 160)
        
        pygame.draw.polygon(screen, blade_color, points)
        pygame.draw.polygon(screen, edge_color, points, 2)
        
        # Guarda pequena
        guard_size = 8
        perpendicular = self.current_angle + math.pi / 2
        guard_x1 = base[0] + math.cos(perpendicular) * guard_size
        guard_y1 = base[1] + math.sin(perpendicular) * guard_size
        guard_x2 = base[0] - math.cos(perpendicular) * guard_size
        guard_y2 = base[1] - math.sin(perpendicular) * guard_size
        pygame.draw.line(screen, (60, 60, 70), 
                        (int(guard_x1), int(guard_y1)), 
                        (int(guard_x2), int(guard_y2)), 3)
        
        # Cabo curto
        handle_length = 10
        handle_x = base[0] - math.cos(self.current_angle) * handle_length
        handle_y = base[1] - math.sin(self.current_angle) * handle_length
        pygame.draw.line(screen, (50, 50, 60), 
                        (int(base[0]), int(base[1])), 
                        (int(handle_x), int(handle_y)), 5)


@WeaponRegistry.register("spear")
class Spear(Weapon):
    """Lança - arma com grande alcance mas área estreita"""
    
    display_name = "Lança"
    description = "Arma de longo alcance, ideal para manter distância."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.thrust_offset = 0
        self.current_angle = 0
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=13,
            attack_cooldown=0.9,
            attack_duration=0.45,
            range=80,
            width=5,
            knockback_force=18,
            critical_chance=0.08,
            critical_multiplier=2.0
        )
    
    def on_attack_start(self):
        self.thrust_offset = -10  # Recua antes de atacar
    
    def update_attack(self, dt: float):
        progress = 1 - (self.attack_timer / self.stats.attack_duration)
        if progress < 0.3:
            # Recuar
            self.thrust_offset = -10 + progress * 10
        elif progress < 0.7:
            # Avançar rápido
            self.thrust_offset = (progress - 0.3) / 0.4 * 30
        else:
            # Voltar
            self.thrust_offset = 30 - (progress - 0.7) / 0.3 * 30
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle
    
    def get_spear_points(self):
        effective_range = self.stats.range * self.owner.stats_manager.get_stats().attack_range
        
        base_distance = self.owner.radius + self.thrust_offset
        base_x = self.owner.x + math.cos(self.current_angle) * base_distance
        base_y = self.owner.y + math.sin(self.current_angle) * base_distance
        
        tip_x = base_x + math.cos(self.current_angle) * effective_range
        tip_y = base_y + math.sin(self.current_angle) * effective_range
        
        # Ponta da lança (triângulo)
        spear_head_length = 15
        head_base_x = tip_x - math.cos(self.current_angle) * spear_head_length
        head_base_y = tip_y - math.sin(self.current_angle) * spear_head_length
        
        perpendicular = self.current_angle + math.pi / 2
        head_width = 8
        
        return {
            'shaft_start': (base_x, base_y),
            'shaft_end': (head_base_x, head_base_y),
            'tip': (tip_x, tip_y),
            'head_left': (head_base_x + math.cos(perpendicular) * head_width,
                         head_base_y + math.sin(perpendicular) * head_width),
            'head_right': (head_base_x - math.cos(perpendicular) * head_width,
                          head_base_y - math.sin(perpendicular) * head_width)
        }
    
    def get_hitbox(self) -> Optional[Dict]:
        points = self.get_spear_points()
        return {
            'type': 'line',
            'start': points['shaft_start'],
            'end': points['tip'],
            'points': [points['head_left'], points['tip'], points['head_right']],
            'active': self.is_attacking,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': self.stats.armor_penetration
        }
    
    def draw(self, screen: pygame.Surface):
        points = self.get_spear_points()
        
        # Cabo (shaft)
        pygame.draw.line(screen, (139, 90, 43),
                        (int(points['shaft_start'][0]), int(points['shaft_start'][1])),
                        (int(points['shaft_end'][0]), int(points['shaft_end'][1])), 4)
        
        # Ponta
        if self.is_attacking:
            head_color = (255, 255, 220)
        else:
            head_color = (200, 200, 210)
        
        head_points = [points['head_left'], points['tip'], points['head_right']]
        pygame.draw.polygon(screen, head_color, head_points)
        pygame.draw.polygon(screen, (150, 150, 160), head_points, 2)


# ============================================================================
# NOVAS ARMAS ESPECIALIZADAS PARA COMBATE EM GRUPO
# ============================================================================

@WeaponRegistry.register("staff")
class HealingStaff(Weapon):
    """Cajado de Cura - arma de suporte que cura aliados"""
    
    display_name = "Cajado de Cura"
    description = "Cura aliados em vez de causar dano. Ideal para suporte."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.current_angle = 0
        self.heal_target = None
        self.heal_beam_active = False
        self.orb_pulse = 0
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=4,
            attack_cooldown=0.5,
            attack_duration=0.35,
            range=130,
            width=15,
            knockback_force=3,
            critical_chance=0.18,
            critical_multiplier=1.4,
            special_effects={'heal_amount': 18}
        )
    
    def on_attack_start(self):
        self.heal_beam_active = True
    
    def on_attack_end(self):
        self.heal_beam_active = False
    
    def update_attack(self, dt: float):
        pass
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle
        self.orb_pulse = (self.orb_pulse + dt * 3) % (math.pi * 2)
    
    def get_heal_info(self) -> Dict:
        """Retorna informações de cura para o sistema de combate"""
        import random
        heal_amount = self.stats.special_effects.get('heal_amount', 15)
        
        # Cura crítica
        if random.random() < self.stats.critical_chance:
            heal_amount *= self.stats.critical_multiplier
        
        return {
            'type': 'heal',
            'amount': heal_amount * self.owner.stats_manager.get_stats().ability_power,
            'range': self.stats.range,
            'active': self.is_attacking
        }
    
    def get_hitbox(self) -> Optional[Dict]:
        # Hitbox menor para dano em inimigos
        effective_range = self.stats.range * 0.5
        
        tip_x = self.owner.x + math.cos(self.current_angle) * (self.owner.radius + effective_range)
        tip_y = self.owner.y + math.sin(self.current_angle) * (self.owner.radius + effective_range)
        
        return {
            'type': 'point',
            'start': (self.owner.x, self.owner.y),
            'end': (tip_x, tip_y),
            'points': [],
            'active': self.is_attacking,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': 0,
            'is_heal_weapon': True,
            'heal_info': self.get_heal_info()
        }
    
    def draw(self, screen: pygame.Surface):
        # Posição do cajado
        staff_length = 45
        base_distance = self.owner.radius + 5
        
        base_x = self.owner.x + math.cos(self.current_angle) * base_distance
        base_y = self.owner.y + math.sin(self.current_angle) * base_distance
        
        tip_x = base_x + math.cos(self.current_angle) * staff_length
        tip_y = base_y + math.sin(self.current_angle) * staff_length
        
        # Cabo do cajado
        pygame.draw.line(screen, (139, 90, 43), 
                        (int(base_x), int(base_y)), 
                        (int(tip_x), int(tip_y)), 5)
        
        # Orbe de cura na ponta
        orb_size = 10 + math.sin(self.orb_pulse) * 2
        if self.is_attacking:
            orb_color = (100, 255, 150)
            # Raio de cura visível
            s = pygame.Surface((self.stats.range * 2, self.stats.range * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (100, 255, 150, 30), 
                             (int(self.stats.range), int(self.stats.range)), int(self.stats.range))
            screen.blit(s, (int(self.owner.x - self.stats.range), int(self.owner.y - self.stats.range)))
        else:
            orb_color = (150, 255, 200)
        
        pygame.draw.circle(screen, orb_color, (int(tip_x), int(tip_y)), int(orb_size))
        pygame.draw.circle(screen, (255, 255, 255), (int(tip_x), int(tip_y)), int(orb_size), 2)


@WeaponRegistry.register("bow")
class Bow(Weapon):
    """Arco - arma de longa distância com projéteis"""
    
    display_name = "Arco"
    description = "Dispara flechas a longa distância. Precisa de mira."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.current_angle = 0
        self.draw_strength = 0  # 0 a 1, quanto mais puxado mais dano
        self.is_drawing = False
        self.arrows: List[Dict] = []  # Projéteis ativos
        self.max_arrows = 5
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=15,
            attack_cooldown=1.0,
            attack_duration=0.25,
            range=220,
            width=3,
            knockback_force=6,
            critical_chance=0.18,
            critical_multiplier=2.2,
            special_effects={'arrow_speed': 16}
        )
    
    def attack(self) -> bool:
        """Dispara uma flecha"""
        if super().attack():
            # Criar flecha
            arrow_speed = self.stats.special_effects.get('arrow_speed', 15)
            self.arrows.append({
                'x': self.owner.x + math.cos(self.current_angle) * (self.owner.radius + 10),
                'y': self.owner.y + math.sin(self.current_angle) * (self.owner.radius + 10),
                'vx': math.cos(self.current_angle) * arrow_speed,
                'vy': math.sin(self.current_angle) * arrow_speed,
                'angle': self.current_angle,
                'damage': self.calculate_damage(),
                'distance_traveled': 0,
                'active': True
            })
            
            # Limitar número de flechas
            if len(self.arrows) > self.max_arrows:
                self.arrows.pop(0)
            
            return True
        return False
    
    def update_attack(self, dt: float):
        pass
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle
        
        # Atualizar flechas
        for arrow in self.arrows[:]:
            if not arrow['active']:
                continue
            
            arrow['x'] += arrow['vx']
            arrow['y'] += arrow['vy']
            arrow['distance_traveled'] += math.sqrt(arrow['vx']**2 + arrow['vy']**2)
            
            # Remover se passou do alcance
            if arrow['distance_traveled'] > self.stats.range:
                arrow['active'] = False
    
    def get_hitbox(self) -> Optional[Dict]:
        # Retorna hitbox das flechas
        active_arrows = [a for a in self.arrows if a['active']]
        return {
            'type': 'projectile',
            'projectiles': active_arrows,
            'active': len(active_arrows) > 0,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': self.stats.armor_penetration
        }
    
    def on_arrow_hit(self, arrow: Dict):
        """Chamado quando uma flecha acerta"""
        arrow['active'] = False
    
    def draw(self, screen: pygame.Surface):
        # Posição do arco
        bow_distance = self.owner.radius + 8
        bow_x = self.owner.x + math.cos(self.current_angle) * bow_distance
        bow_y = self.owner.y + math.sin(self.current_angle) * bow_distance
        
        # Desenhar arco (curva)
        perpendicular = self.current_angle + math.pi / 2
        bow_width = 20
        
        top_x = bow_x + math.cos(perpendicular) * bow_width
        top_y = bow_y + math.sin(perpendicular) * bow_width
        bottom_x = bow_x - math.cos(perpendicular) * bow_width
        bottom_y = bow_y - math.sin(perpendicular) * bow_width
        
        # Curva do arco
        mid_x = bow_x + math.cos(self.current_angle) * 8
        mid_y = bow_y + math.sin(self.current_angle) * 8
        
        pygame.draw.line(screen, (139, 90, 43), (int(top_x), int(top_y)), (int(mid_x), int(mid_y)), 3)
        pygame.draw.line(screen, (139, 90, 43), (int(mid_x), int(mid_y)), (int(bottom_x), int(bottom_y)), 3)
        
        # Corda
        string_back = bow_x - math.cos(self.current_angle) * 5
        string_back_y = bow_y - math.sin(self.current_angle) * 5
        pygame.draw.line(screen, (200, 200, 200), (int(top_x), int(top_y)), (int(string_back), int(string_back_y)), 1)
        pygame.draw.line(screen, (200, 200, 200), (int(string_back), int(string_back_y)), (int(bottom_x), int(bottom_y)), 1)
        
        # Desenhar flechas
        for arrow in self.arrows:
            if arrow['active']:
                # Corpo da flecha
                arrow_length = 15
                back_x = arrow['x'] - math.cos(arrow['angle']) * arrow_length
                back_y = arrow['y'] - math.sin(arrow['angle']) * arrow_length
                
                pygame.draw.line(screen, (139, 90, 43), 
                               (int(back_x), int(back_y)), 
                               (int(arrow['x']), int(arrow['y'])), 2)
                
                # Ponta da flecha
                pygame.draw.circle(screen, (200, 200, 210), 
                                 (int(arrow['x']), int(arrow['y'])), 3)


@WeaponRegistry.register("warhammer")
class Warhammer(Weapon):
    """Martelo de Guerra - arma pesada que causa stun"""
    
    display_name = "Martelo de Guerra"
    description = "Arma pesada e lenta que stuna inimigos ao acertar."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.swing_angle = 0
        self.swing_range = math.pi * 0.8
        self.current_angle = 0
        self.slam_effect = False
        self.slam_timer = 0
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=22,
            attack_cooldown=1.7,
            attack_duration=0.85,
            range=55,
            width=25,
            knockback_force=30,
            critical_chance=0.08,
            critical_multiplier=1.8,
            special_effects={'stun_duration': 1.0}
        )
    
    def on_attack_start(self):
        self.swing_angle = -self.swing_range / 2
    
    def update_attack(self, dt: float):
        progress = 1 - (self.attack_timer / self.stats.attack_duration)
        
        # Movimento de slam (levanta e desce)
        if progress < 0.4:
            # Levantar
            self.swing_angle = -self.swing_range / 2 + (progress / 0.4) * (self.swing_range / 2)
        elif progress < 0.6:
            # Manter no alto
            self.swing_angle = 0
        else:
            # Slam para baixo
            self.swing_angle = (progress - 0.6) / 0.4 * (self.swing_range / 2)
            if progress > 0.85 and not self.slam_effect:
                self.slam_effect = True
                self.slam_timer = 0.3
    
    def on_attack_end(self):
        self.slam_effect = False
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle + self.swing_angle
        
        if self.slam_timer > 0:
            self.slam_timer -= dt
    
    def get_stun_effect(self) -> StatusEffect:
        """Retorna o efeito de stun para aplicar"""
        return StatusEffect(
            name='warhammer_stun',
            effect_type=StatusEffectType.STUN,
            duration=self.stats.special_effects.get('stun_duration', 0.8),
            source=self.owner
        )
    
    def get_hitbox(self) -> Optional[Dict]:
        effective_range = self.stats.range * self.owner.stats_manager.get_stats().attack_range
        
        head_x = self.owner.x + math.cos(self.current_angle) * (self.owner.radius + effective_range)
        head_y = self.owner.y + math.sin(self.current_angle) * (self.owner.radius + effective_range)
        
        return {
            'type': 'circle',
            'center': (head_x, head_y),
            'radius': self.stats.width,
            'active': self.is_attacking and self.attack_timer < self.stats.attack_duration * 0.4,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': self.stats.armor_penetration,
            'applies_stun': True,
            'stun_effect': self.get_stun_effect()
        }
    
    def draw(self, screen: pygame.Surface):
        effective_range = self.stats.range * self.owner.stats_manager.get_stats().attack_range
        
        # Cabo
        base_distance = self.owner.radius + 5
        base_x = self.owner.x + math.cos(self.current_angle) * base_distance
        base_y = self.owner.y + math.sin(self.current_angle) * base_distance
        
        head_x = self.owner.x + math.cos(self.current_angle) * (self.owner.radius + effective_range)
        head_y = self.owner.y + math.sin(self.current_angle) * (self.owner.radius + effective_range)
        
        pygame.draw.line(screen, (100, 80, 60), 
                        (int(base_x), int(base_y)), 
                        (int(head_x), int(head_y)), 6)
        
        # Cabeça do martelo
        if self.is_attacking:
            head_color = (255, 200, 100)
        else:
            head_color = (150, 150, 160)
        
        # Desenhar cabeça retangular do martelo
        perpendicular = self.current_angle + math.pi / 2
        half_width = self.stats.width / 2
        
        hammer_points = [
            (head_x + math.cos(perpendicular) * half_width - math.cos(self.current_angle) * 8,
             head_y + math.sin(perpendicular) * half_width - math.sin(self.current_angle) * 8),
            (head_x + math.cos(perpendicular) * half_width + math.cos(self.current_angle) * 8,
             head_y + math.sin(perpendicular) * half_width + math.sin(self.current_angle) * 8),
            (head_x - math.cos(perpendicular) * half_width + math.cos(self.current_angle) * 8,
             head_y - math.sin(perpendicular) * half_width + math.sin(self.current_angle) * 8),
            (head_x - math.cos(perpendicular) * half_width - math.cos(self.current_angle) * 8,
             head_y - math.sin(perpendicular) * half_width - math.sin(self.current_angle) * 8),
        ]
        
        pygame.draw.polygon(screen, head_color, hammer_points)
        pygame.draw.polygon(screen, (100, 100, 110), hammer_points, 2)
        
        # Efeito de slam
        if self.slam_timer > 0:
            alpha = int(150 * (self.slam_timer / 0.3))
            s = pygame.Surface((self.stats.width * 4, self.stats.width * 4), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 200, 100, alpha), 
                             (self.stats.width * 2, self.stats.width * 2), 
                             int(self.stats.width * 2 * (1 - self.slam_timer / 0.3)))
            screen.blit(s, (int(head_x - self.stats.width * 2), int(head_y - self.stats.width * 2)))


@WeaponRegistry.register("tome")
class MagicTome(Weapon):
    """Tomo Mágico - arma de suporte que aplica buffs"""
    
    display_name = "Tomo Mágico"
    description = "Livro de magias que fortalece aliados próximos."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.current_angle = 0
        self.page_flutter = 0
        self.buff_pulse = 0
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=6,
            attack_cooldown=0.9,
            attack_duration=0.45,
            range=110,
            width=10,
            knockback_force=2,
            critical_chance=0.1,
            critical_multiplier=1.4,
            special_effects={
                'buff_damage': 0.20,
                'buff_duration': 5.0
            }
        )
    
    def on_attack_start(self):
        self.page_flutter = 0
    
    def update_attack(self, dt: float):
        self.page_flutter += dt * 20
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle
        self.buff_pulse = (self.buff_pulse + dt * 2) % (math.pi * 2)
    
    def get_buff_effect(self) -> StatusEffect:
        """Retorna o efeito de buff para aplicar a aliados"""
        return StatusEffect(
            name='tome_damage_buff',
            effect_type=StatusEffectType.BUFF_DAMAGE,
            duration=self.stats.special_effects.get('buff_duration', 4.0),
            power=self.stats.special_effects.get('buff_damage', 0.15),
            source=self.owner
        )
    
    def get_hitbox(self) -> Optional[Dict]:
        # Projétil mágico para dano
        tip_x = self.owner.x + math.cos(self.current_angle) * (self.owner.radius + 30)
        tip_y = self.owner.y + math.sin(self.current_angle) * (self.owner.radius + 30)
        
        return {
            'type': 'point',
            'start': (self.owner.x, self.owner.y),
            'end': (tip_x, tip_y),
            'points': [],
            'active': self.is_attacking,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': 0.2,  # Magia ignora um pouco de armadura
            'is_buff_weapon': True,
            'buff_effect': self.get_buff_effect(),
            'buff_range': self.stats.range
        }
    
    def draw(self, screen: pygame.Surface):
        # Posição do tomo
        tome_distance = self.owner.radius + 10
        tome_x = self.owner.x + math.cos(self.current_angle) * tome_distance
        tome_y = self.owner.y + math.sin(self.current_angle) * tome_distance
        
        # Desenhar livro
        book_width = 15
        book_height = 20
        
        perpendicular = self.current_angle + math.pi / 2
        
        # Capa do livro
        if self.is_attacking:
            book_color = (200, 150, 255)
            # Aura de buff
            pulse_size = 5 + math.sin(self.buff_pulse) * 3
            s = pygame.Surface((self.stats.range * 2, self.stats.range * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (200, 150, 255, 30), 
                             (int(self.stats.range), int(self.stats.range)), int(self.stats.range))
            screen.blit(s, (int(self.owner.x - self.stats.range), int(self.owner.y - self.stats.range)))
        else:
            book_color = (150, 100, 200)
        
        # Páginas (flutuando se atacando)
        page_offset = math.sin(self.page_flutter) * 3 if self.is_attacking else 0
        
        book_points = [
            (tome_x + math.cos(perpendicular) * book_width/2 + page_offset,
             tome_y + math.sin(perpendicular) * book_width/2),
            (tome_x - math.cos(perpendicular) * book_width/2 + page_offset,
             tome_y - math.sin(perpendicular) * book_width/2),
            (tome_x - math.cos(perpendicular) * book_width/2 + math.cos(self.current_angle) * book_height,
             tome_y - math.sin(perpendicular) * book_width/2 + math.sin(self.current_angle) * book_height),
            (tome_x + math.cos(perpendicular) * book_width/2 + math.cos(self.current_angle) * book_height,
             tome_y + math.sin(perpendicular) * book_width/2 + math.sin(self.current_angle) * book_height),
        ]
        
        pygame.draw.polygon(screen, book_color, book_points)
        pygame.draw.polygon(screen, (100, 70, 150), book_points, 2)
        
        # Símbolo mágico no centro
        center_x = tome_x + math.cos(self.current_angle) * book_height/2
        center_y = tome_y + math.sin(self.current_angle) * book_height/2
        pygame.draw.circle(screen, (255, 220, 150), (int(center_x), int(center_y)), 4)


@WeaponRegistry.register("shield_bash")
class ShieldBash(Weapon):
    """Escudo - arma defensiva que também pode atacar"""
    
    display_name = "Escudo"
    description = "Arma defensiva que pode bloquear e dar bash."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.current_angle = 0
        self.bash_offset = 0
        self.is_blocking = False
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=10,
            attack_cooldown=0.9,
            attack_duration=0.35,
            range=35,
            width=32,
            knockback_force=25,
            critical_chance=0.05,
            critical_multiplier=1.3,
            special_effects={'slow_duration': 1.2, 'slow_power': 0.35}
        )
    
    def on_attack_start(self):
        self.bash_offset = -5
    
    def update_attack(self, dt: float):
        progress = 1 - (self.attack_timer / self.stats.attack_duration)
        if progress < 0.5:
            self.bash_offset = -5 + progress * 2 * 25
        else:
            self.bash_offset = 20 - (progress - 0.5) * 2 * 25
    
    def on_attack_end(self):
        self.bash_offset = 0
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle
    
    def get_slow_effect(self) -> StatusEffect:
        """Retorna efeito de slow para aplicar"""
        return StatusEffect(
            name='shield_bash_slow',
            effect_type=StatusEffectType.SLOW,
            duration=self.stats.special_effects.get('slow_duration', 1.0),
            power=self.stats.special_effects.get('slow_power', 0.3),
            source=self.owner
        )
    
    def get_hitbox(self) -> Optional[Dict]:
        effective_range = self.stats.range + self.bash_offset
        
        center_x = self.owner.x + math.cos(self.current_angle) * (self.owner.radius + effective_range/2)
        center_y = self.owner.y + math.sin(self.current_angle) * (self.owner.radius + effective_range/2)
        
        return {
            'type': 'circle',
            'center': (center_x, center_y),
            'radius': self.stats.width / 2,
            'active': self.is_attacking,
            'damage': self.calculate_damage(),
            'knockback': self.stats.knockback_force,
            'armor_penetration': 0,
            'applies_slow': True,
            'slow_effect': self.get_slow_effect()
        }
    
    def draw(self, screen: pygame.Surface):
        effective_range = self.stats.range + self.bash_offset
        
        # Centro do escudo
        shield_x = self.owner.x + math.cos(self.current_angle) * (self.owner.radius + effective_range/2)
        shield_y = self.owner.y + math.sin(self.current_angle) * (self.owner.radius + effective_range/2)
        
        # Desenhar escudo (elipse)
        perpendicular = self.current_angle + math.pi / 2
        half_width = self.stats.width / 2
        half_height = self.stats.width / 3
        
        if self.is_attacking:
            shield_color = (200, 200, 255)
        else:
            shield_color = (150, 150, 180)
        
        # Pontos do escudo (forma de escudo medieval)
        shield_points = [
            (shield_x + math.cos(perpendicular) * half_width,
             shield_y + math.sin(perpendicular) * half_width),
            (shield_x + math.cos(perpendicular) * half_width * 0.7 + math.cos(self.current_angle) * half_height,
             shield_y + math.sin(perpendicular) * half_width * 0.7 + math.sin(self.current_angle) * half_height),
            (shield_x + math.cos(self.current_angle) * half_height * 1.5,
             shield_y + math.sin(self.current_angle) * half_height * 1.5),
            (shield_x - math.cos(perpendicular) * half_width * 0.7 + math.cos(self.current_angle) * half_height,
             shield_y - math.sin(perpendicular) * half_width * 0.7 + math.sin(self.current_angle) * half_height),
            (shield_x - math.cos(perpendicular) * half_width,
             shield_y - math.sin(perpendicular) * half_width),
        ]
        
        pygame.draw.polygon(screen, shield_color, shield_points)
        pygame.draw.polygon(screen, (100, 100, 130), shield_points, 2)
        
        # Cruz no escudo
        pygame.draw.line(screen, (200, 180, 100), 
                        (int(shield_x - math.cos(perpendicular) * half_width * 0.4),
                         int(shield_y - math.sin(perpendicular) * half_width * 0.4)),
                        (int(shield_x + math.cos(perpendicular) * half_width * 0.4),
                         int(shield_y + math.sin(perpendicular) * half_width * 0.4)), 3)


@WeaponRegistry.register("trap_launcher")
class TrapLauncher(Weapon):
    """Lançador de Armadilhas - lança armadilhas à distância que enraízam inimigos"""
    
    display_name = "Lançador de Armadilhas"
    description = "Lança armadilhas à distância que enraízam e causam dano."
    
    def __init__(self, owner: 'Entity'):
        super().__init__(owner)
        self.current_angle = 0
        self.launch_animation = 0
        self.launched_traps: List[Dict] = []  # Armadilhas em voo
        self.ground_traps: List[Dict] = []    # Armadilhas no chão
        self.max_ground_traps = 4
        self.trap_duration = 12.0
    
    def get_default_stats(self) -> WeaponStats:
        return WeaponStats(
            base_damage=15,
            attack_cooldown=1.0,
            attack_duration=0.35,
            range=180,
            width=28,
            knockback_force=4,
            critical_chance=0.08,
            critical_multiplier=1.4,
            special_effects={
                'trap_damage': 25,
                'root_duration': 2.0,
                'projectile_speed': 380
            }
        )
    
    def on_attack_start(self):
        self.launch_animation = 0
        # Lançar armadilha
        if len(self.ground_traps) < self.max_ground_traps:
            trap = {
                'x': self.owner.x + math.cos(self.current_angle) * (self.owner.radius + 15),
                'y': self.owner.y + math.sin(self.current_angle) * (self.owner.radius + 15),
                'vx': math.cos(self.current_angle) * self.stats.special_effects['projectile_speed'],
                'vy': math.sin(self.current_angle) * self.stats.special_effects['projectile_speed'],
                'target_distance': self.stats.range * 0.8,  # Distância alvo
                'traveled': 0,
                'active': True,
                'in_flight': True
            }
            self.launched_traps.append(trap)
    
    def update_attack(self, dt: float):
        self.launch_animation += dt * 5
    
    def update(self, dt: float):
        super().update(dt)
        self.current_angle = self.owner.facing_angle
        
        # Atualizar armadilhas em voo
        for trap in self.launched_traps[:]:
            if trap['in_flight']:
                # Mover
                trap['x'] += trap['vx'] * dt
                trap['y'] += trap['vy'] * dt
                trap['traveled'] += abs(trap['vx'] * dt) + abs(trap['vy'] * dt)
                
                # Verificar se chegou ao destino
                if trap['traveled'] >= trap['target_distance']:
                    trap['in_flight'] = False
                    # Converter para armadilha no chão
                    self.ground_traps.append({
                        'x': trap['x'],
                        'y': trap['y'],
                        'radius': self.stats.width,
                        'duration': self.trap_duration,
                        'damage': self.stats.special_effects['trap_damage'],
                        'root_duration': self.stats.special_effects['root_duration'],
                        'active': True,
                        'armed_time': 0.3  # Tempo para armar
                    })
                    self.launched_traps.remove(trap)
        
        # Atualizar armadilhas no chão
        for trap in self.ground_traps[:]:
            trap['duration'] -= dt
            if trap.get('armed_time', 0) > 0:
                trap['armed_time'] -= dt
            
            if trap['duration'] <= 0 or not trap['active']:
                self.ground_traps.remove(trap)
    
    def check_trap_hits(self, entities: List) -> List[tuple]:
        """Verifica se inimigos pisaram nas armadilhas. Retorna lista de (entidade, trap)"""
        hits = []
        for trap in self.ground_traps[:]:
            if not trap['active'] or trap.get('armed_time', 0) > 0:
                continue
            
            for entity in entities:
                if entity == self.owner or not entity.is_alive():
                    continue
                # Verificar se é inimigo
                if self.owner.team != "none" and entity.team == self.owner.team:
                    continue
                    
                dx = entity.x - trap['x']
                dy = entity.y - trap['y']
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist <= trap['radius'] + entity.radius:
                    hits.append((entity, trap))
                    trap['active'] = False
        
        return hits
    
    def get_hitbox(self) -> Optional[Dict]:
        # Armadilhas em voo podem causar dano
        active_projectiles = [t for t in self.launched_traps if t['in_flight']]
        
        return {
            'type': 'projectile',
            'projectiles': [
                {'x': t['x'], 'y': t['y'], 'radius': 8, 'active': True}
                for t in active_projectiles
            ],
            'active': len(active_projectiles) > 0,
            'damage': self.stats.base_damage * 0.5,  # Dano reduzido se acertar em voo
            'knockback': self.stats.knockback_force,
            'armor_penetration': 0
        }
    
    def on_hit(self, target, damage: float):
        """Chamado quando armadilha em voo acerta alguém"""
        pass  # Armadilha em voo só causa dano leve
    
    def draw(self, screen: pygame.Surface):
        # Posição do lançador
        launcher_distance = self.owner.radius + 10
        launcher_x = self.owner.x + math.cos(self.current_angle) * launcher_distance
        launcher_y = self.owner.y + math.sin(self.current_angle) * launcher_distance
        
        # Corpo do lançador (cilindro)
        pygame.draw.circle(screen, (100, 80, 60), (int(launcher_x), int(launcher_y)), 10)
        
        # Tubo
        tube_length = 25
        tip_x = launcher_x + math.cos(self.current_angle) * tube_length
        tip_y = launcher_y + math.sin(self.current_angle) * tube_length
        
        if self.is_attacking:
            tube_color = (180, 140, 80)
        else:
            tube_color = (120, 100, 70)
        
        pygame.draw.line(screen, tube_color, 
                        (int(launcher_x), int(launcher_y)), 
                        (int(tip_x), int(tip_y)), 6)
        
        # Ponta do tubo
        pygame.draw.circle(screen, (80, 60, 40), (int(tip_x), int(tip_y)), 5)
        
        # Desenhar armadilhas em voo
        for trap in self.launched_traps:
            if trap['in_flight']:
                pygame.draw.circle(screen, (200, 150, 50), 
                                 (int(trap['x']), int(trap['y'])), 6)
                pygame.draw.circle(screen, (150, 100, 30), 
                                 (int(trap['x']), int(trap['y'])), 6, 2)
        
        # Desenhar armadilhas no chão
        for trap in self.ground_traps:
            if trap['active']:
                # Cor baseada no estado de armado
                if trap.get('armed_time', 0) > 0:
                    # Ainda armando - amarelo piscando
                    alpha = int(128 + 127 * math.sin(trap['armed_time'] * 20))
                    trap_color = (200, 200, 50)
                else:
                    # Armada - marrom/laranja
                    trap_color = (180, 120, 40)
                
                # Círculo da armadilha
                pygame.draw.circle(screen, trap_color, 
                                 (int(trap['x']), int(trap['y'])), int(trap['radius']), 2)
                
                # Dentes da armadilha (visual)
                num_teeth = 8
                for i in range(num_teeth):
                    angle = (2 * math.pi * i) / num_teeth
                    inner_r = trap['radius'] * 0.6
                    outer_r = trap['radius'] * 0.9
                    
                    x1 = trap['x'] + math.cos(angle) * inner_r
                    y1 = trap['y'] + math.sin(angle) * inner_r
                    x2 = trap['x'] + math.cos(angle) * outer_r
                    y2 = trap['y'] + math.sin(angle) * outer_r
                    
                    pygame.draw.line(screen, trap_color, 
                                   (int(x1), int(y1)), (int(x2), int(y2)), 2)
                
                # Centro
                pygame.draw.circle(screen, (220, 160, 60), 
                                 (int(trap['x']), int(trap['y'])), 4)

