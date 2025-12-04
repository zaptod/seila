"""
Sistema de Entidades e Classes
==============================
Sistema modular para entidades (círculos) com registro automático de classes.
Inclui classes de combate, suporte, controle e defesa para modo torneio.
"""

import pygame
import math
from abc import ABC, abstractmethod
from typing import Dict, Type, List, Optional, TYPE_CHECKING, Tuple
from stats import BaseStats, StatsManager, StatusEffectManager, StatusEffect, StatusEffectType
from weapons import WeaponRegistry, Weapon

if TYPE_CHECKING:
    from controller import Controller


# ============================================================================
# REGISTRO DE CLASSES
# ============================================================================

class ClassRegistry:
    """Registro global de todas as classes de personagem disponíveis"""
    _classes: Dict[str, Type['Entity']] = {}
    
    @classmethod
    def register(cls, class_id: str):
        """Decorator para registrar uma classe"""
        def decorator(entity_class: Type['Entity']):
            cls._classes[class_id] = entity_class
            entity_class.class_id = class_id
            return entity_class
        return decorator
    
    @classmethod
    def create(cls, class_id: str, x: float, y: float, color: tuple) -> 'Entity':
        """Cria uma instância de entidade pelo ID da classe"""
        if class_id not in cls._classes:
            raise ValueError(f"Classe não encontrada: {class_id}")
        return cls._classes[class_id](x, y, color)
    
    @classmethod
    def get_all(cls) -> Dict[str, Type['Entity']]:
        """Retorna todas as classes registradas"""
        return cls._classes.copy()
    
    @classmethod
    def list_classes(cls) -> List[str]:
        """Lista IDs de todas as classes"""
        return list(cls._classes.keys())


# ============================================================================
# CLASSE BASE DE ENTIDADE
# ============================================================================

class Entity(ABC):
    """Classe base abstrata para todas as entidades do jogo"""
    
    class_id: str = "base"
    display_name: str = "Entidade Base"
    description: str = "Uma entidade básica"
    default_weapon: str = "sword"
    role: str = "damage"  # Role: 'damage', 'tank', 'support', 'control', 'ranged'
    
    def __init__(self, x: float, y: float, color: tuple):
        self.x = x
        self.y = y
        self.color = color
        self.vx = 0
        self.vy = 0
        
        # Sistema de stats
        self.stats_manager = StatsManager(self.get_default_stats())
        
        # Sistema de status effects
        self.status_effects = StatusEffectManager(self)
        
        # Atalhos para stats comuns
        self._update_cached_stats()
        
        # Saúde atual
        self.health = self.stats_manager.get_stats().max_health
        
        # Invulnerabilidade
        self.invulnerable_time = 0
        self.invulnerable_duration = 0.5
        
        # Direção
        self.facing_angle = 0
        self.rotation_speed = 8.0  # Velocidade de rotação (radianos por segundo)
        
        # Arma
        self.weapon: Optional[Weapon] = None
        self.set_weapon(self.default_weapon)
        
        # Controller (será definido externamente)
        self.controller: Optional['Controller'] = None
        
        # Habilidade
        self.ability_cooldown = 0
        
        # Aliados (para habilidades de suporte)
        self.allies: List['Entity'] = []
        
        # Team (para batalhas em grupo)
        self.team: str = "none"  # "blue", "red", ou "none"
        
        # Estado para IA
        self.moving = False
        self.last_damage_source = None
    
    def _update_cached_stats(self):
        """Atualiza cache de stats comuns para acesso rápido"""
        stats = self.stats_manager.get_stats()
        self.radius = stats.radius
        self.mass = stats.mass
        self.speed = stats.speed
        self.friction = stats.friction
    
    @abstractmethod
    def get_default_stats(self) -> BaseStats:
        """Retorna os stats padrão da classe"""
        pass
    
    def get_ability_info(self) -> Dict:
        """Retorna informações sobre a habilidade da classe"""
        return {
            'name': 'Nenhuma',
            'description': 'Sem habilidade',
            'cooldown': 0
        }
    
    def use_ability(self) -> bool:
        """Usa a habilidade da classe. Retorna True se usou."""
        return False
    
    def set_weapon(self, weapon_id: str):
        """Define a arma da entidade"""
        self.weapon = WeaponRegistry.create(weapon_id, self)
    
    def set_controller(self, controller: 'Controller'):
        """Define o controlador (Player ou IA)"""
        self.controller = controller
        controller.entity = self
    
    def set_allies(self, allies: List['Entity']):
        """Define a lista de aliados"""
        self.allies = [a for a in allies if a != self]
    
    def is_alive(self) -> bool:
        return self.health > 0
    
    def can_move(self) -> bool:
        """Verifica se pode se mover (não stunado/rooted)"""
        return not self.status_effects.is_rooted()
    
    def can_attack(self) -> bool:
        """Verifica se pode atacar (não stunado)"""
        return not self.status_effects.is_stunned()
    
    def can_use_ability(self) -> bool:
        """Verifica se pode usar habilidade (não stunado/silenciado)"""
        return not self.status_effects.is_silenced() and self.ability_cooldown <= 0
    
    def take_damage(self, damage: float, source: 'Entity' = None, armor_penetration: float = 0) -> bool:
        """Recebe dano. Retorna True se o dano foi aplicado."""
        if self.invulnerable_time > 0:
            return False
        
        # Verificar invulnerabilidade por status
        if self.status_effects.has_effect(StatusEffectType.INVULNERABLE):
            return False
        
        # Calcular dano final
        final_damage = self.stats_manager.calculate_damage_taken(damage, armor_penetration)
        
        # Modificador de dano por status (marked, etc)
        final_damage *= self.status_effects.get_damage_multiplier()
        
        # Absorver com escudo primeiro
        final_damage = self.status_effects.absorb_damage(final_damage)
        
        if final_damage > 0:
            self.health -= final_damage
            self.invulnerable_time = self.invulnerable_duration
            self.last_damage_source = source
            return True
        
        return False
    
    def heal(self, amount: float):
        """Cura a entidade"""
        stats = self.stats_manager.get_stats()
        self.health = min(stats.max_health, self.health + amount)
    
    def apply_knockback(self, force: float, angle: float):
        """Aplica knockback na entidade"""
        # Não aplica knockback se rooted/stunned
        if self.status_effects.is_rooted():
            return
        effective_force = self.stats_manager.calculate_knockback(force)
        self.vx += math.cos(angle) * effective_force
        self.vy += math.sin(angle) * effective_force
    
    def apply_status_effect(self, effect: StatusEffect) -> bool:
        """Aplica um efeito de status"""
        return self.status_effects.apply_effect(effect)
    
    def update(self, dt: float):
        """Atualiza a entidade"""
        # Atualizar stats manager (buffs/debuffs temporários)
        self.stats_manager.update(dt)
        self._update_cached_stats()
        
        # Atualizar status effects
        self.status_effects.update(dt)
        
        # Regeneração de vida
        stats = self.stats_manager.get_stats()
        if stats.health_regen > 0 and self.health < stats.max_health:
            self.heal(stats.health_regen * dt)
        
        # Atualizar invulnerabilidade
        if self.invulnerable_time > 0:
            self.invulnerable_time -= dt
        
        # Atualizar cooldown de habilidade
        if self.ability_cooldown > 0:
            self.ability_cooldown -= dt
        
        # Processar input do controller
        if self.controller:
            self.controller.update(dt)
        
        # Aplicar velocidade
        self.x += self.vx * dt * 60
        self.y += self.vy * dt * 60
        
        # Aplicar fricção
        self.vx *= self.friction
        self.vy *= self.friction
        
        if abs(self.vx) < 0.1:
            self.vx = 0
        if abs(self.vy) < 0.1:
            self.vy = 0
        
        # Atualizar arma
        if self.weapon:
            self.weapon.update(dt)
    
    def move(self, dx: float, dy: float):
        """Move a entidade na direção especificada"""
        # Não pode mover se rooted/stunned
        if not self.can_move():
            return
        
        # Normalizar diagonal
        if dx != 0 and dy != 0:
            length = math.sqrt(dx * dx + dy * dy)
            dx /= length
            dy /= length
        
        stats = self.stats_manager.get_stats()
        
        # Aplicar slow de status effects
        slow_factor = self.status_effects.get_slow_factor()
        effective_speed = stats.speed * slow_factor
        
        self.vx += dx * effective_speed * stats.acceleration
        self.vy += dy * effective_speed * stats.acceleration
        
        # Atualizar direção com rotação suave
        if dx != 0 or dy != 0:
            target_angle = math.atan2(dy, dx)
            # Calcular a diferença de ângulo (menor caminho)
            angle_diff = target_angle - self.facing_angle
            # Normalizar para -pi a pi
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            # Aplicar rotação suave (usando dt fixo de 1/60)
            max_rotation = self.rotation_speed * (1/60)
            if abs(angle_diff) <= max_rotation:
                self.facing_angle = target_angle
            else:
                self.facing_angle += max_rotation if angle_diff > 0 else -max_rotation
            # Normalizar facing_angle
            while self.facing_angle > math.pi:
                self.facing_angle -= 2 * math.pi
            while self.facing_angle < -math.pi:
                self.facing_angle += 2 * math.pi
            self.moving = True
    
    def attack(self) -> bool:
        """Ataca com a arma. Retorna True se atacou."""
        if not self.can_attack():
            return False
        if self.weapon:
            return self.weapon.attack()
        return False
    
    def draw(self, screen: pygame.Surface):
        """Desenha a entidade"""
        # Piscar se invulnerável
        if self.invulnerable_time > 0 and int(self.invulnerable_time * 10) % 2 == 0:
            color = tuple(min(255, c + 100) for c in self.color)
        else:
            color = self.color
        
        # Efeito visual de status
        self._draw_status_effects_visual(screen)
        
        # Corpo
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), int(self.radius))
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), int(self.radius), 2)
        
        # Indicador de direção
        indicator_length = self.radius + 10
        end_x = self.x + math.cos(self.facing_angle) * indicator_length
        end_y = self.y + math.sin(self.facing_angle) * indicator_length
        pygame.draw.line(screen, (255, 255, 255), 
                        (int(self.x), int(self.y)), 
                        (int(end_x), int(end_y)), 2)
        
        # Arma
        if self.weapon:
            self.weapon.draw(screen)
        
        # Barra de vida
        self._draw_health_bar(screen)
        
        # Barra de escudo (se tiver)
        if self.status_effects.shield_amount > 0:
            self._draw_shield_bar(screen)
    
    def _draw_status_effects_visual(self, screen: pygame.Surface):
        """Desenha efeitos visuais para status effects ativos"""
        # Stun - estrelas girando
        if self.status_effects.is_stunned():
            # Círculo amarelo pulsante
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.01)) * 0.3 + 0.7
            s = pygame.Surface((int(self.radius * 3), int(self.radius * 3)), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 255, 0, int(100 * pulse)), 
                             (int(self.radius * 1.5), int(self.radius * 1.5)), int(self.radius + 5), 3)
            screen.blit(s, (int(self.x - self.radius * 1.5), int(self.y - self.radius * 1.5)))
        
        # Root - correntes verdes no chão
        if self.status_effects.has_effect(StatusEffectType.ROOT) and not self.status_effects.is_stunned():
            pygame.draw.circle(screen, (100, 200, 100), (int(self.x), int(self.y)), int(self.radius + 8), 2)
        
        # Slow - aura azul
        if self.status_effects.has_effect(StatusEffectType.SLOW):
            s = pygame.Surface((int(self.radius * 3), int(self.radius * 3)), pygame.SRCALPHA)
            pygame.draw.circle(s, (100, 150, 255, 50), 
                             (int(self.radius * 1.5), int(self.radius * 1.5)), int(self.radius + 3))
            screen.blit(s, (int(self.x - self.radius * 1.5), int(self.y - self.radius * 1.5)))
        
        # Shield - círculo ciano
        if self.status_effects.shield_amount > 0:
            pygame.draw.circle(screen, (100, 200, 255), (int(self.x), int(self.y)), int(self.radius + 6), 2)
        
        # Burn - partículas de fogo
        if self.status_effects.has_effect(StatusEffectType.BURN):
            for i in range(3):
                offset = math.sin(pygame.time.get_ticks() * 0.01 + i) * 5
                pygame.draw.circle(screen, (255, 150, 50), 
                                  (int(self.x + offset), int(self.y - self.radius - 5 - i * 5)), 3)
        
        # Poison - aura verde
        if self.status_effects.has_effect(StatusEffectType.POISON):
            s = pygame.Surface((int(self.radius * 3), int(self.radius * 3)), pygame.SRCALPHA)
            pygame.draw.circle(s, (100, 255, 100, 40), 
                             (int(self.radius * 1.5), int(self.radius * 1.5)), int(self.radius + 4))
            screen.blit(s, (int(self.x - self.radius * 1.5), int(self.y - self.radius * 1.5)))
        
        # Heal over time - partículas verdes subindo
        if self.status_effects.has_effect(StatusEffectType.HEAL_OVER_TIME):
            t = pygame.time.get_ticks() * 0.005
            for i in range(2):
                offset_x = math.sin(t + i * 2) * 10
                offset_y = (t * 20 + i * 15) % 30
                pygame.draw.circle(screen, (100, 255, 150), 
                                  (int(self.x + offset_x), int(self.y - offset_y)), 3)
        
        # Marked - X vermelho acima
        if self.status_effects.has_effect(StatusEffectType.MARKED):
            mark_y = self.y - self.radius - 25
            pygame.draw.line(screen, (255, 50, 50), 
                           (int(self.x - 6), int(mark_y - 6)), (int(self.x + 6), int(mark_y + 6)), 2)
            pygame.draw.line(screen, (255, 50, 50), 
                           (int(self.x + 6), int(mark_y - 6)), (int(self.x - 6), int(mark_y + 6)), 2)
        
        # Buff de dano - setas vermelhas para cima
        if self.status_effects.has_effect(StatusEffectType.BUFF_DAMAGE):
            pygame.draw.polygon(screen, (255, 100, 100), [
                (int(self.x), int(self.y - self.radius - 20)),
                (int(self.x - 5), int(self.y - self.radius - 15)),
                (int(self.x + 5), int(self.y - self.radius - 15))
            ])
    
    def _draw_shield_bar(self, screen: pygame.Surface):
        """Desenha a barra de escudo"""
        bar_width = 50
        bar_height = 4
        bar_x = self.x - bar_width // 2
        bar_y = self.y - self.radius - 22  # Acima da barra de vida
        
        stats = self.stats_manager.get_stats()
        # Shield ratio baseado na vida máxima (escudo pode ser maior que vida)
        shield_ratio = min(1.0, self.status_effects.shield_amount / (stats.max_health * 0.5))
        
        pygame.draw.rect(screen, (100, 200, 255), 
                        (bar_x, bar_y, bar_width * shield_ratio, bar_height))
        pygame.draw.rect(screen, (150, 220, 255), 
                        (bar_x, bar_y, bar_width, bar_height), 1)
    
    def _draw_health_bar(self, screen: pygame.Surface):
        """Desenha a barra de vida"""
        stats = self.stats_manager.get_stats()
        bar_width = 50
        bar_height = 6
        bar_x = self.x - bar_width // 2
        bar_y = self.y - self.radius - 15
        
        health_ratio = self.health / stats.max_health
        
        # Fundo
        pygame.draw.rect(screen, (60, 60, 60), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Cor baseada na vida
        if health_ratio > 0.5:
            health_color = (100, 255, 100)
        elif health_ratio > 0.25:
            health_color = (255, 255, 100)
        else:
            health_color = (255, 100, 100)
        
        pygame.draw.rect(screen, health_color, 
                        (bar_x, bar_y, bar_width * health_ratio, bar_height))
        pygame.draw.rect(screen, (255, 255, 255), 
                        (bar_x, bar_y, bar_width, bar_height), 1)
    
    def get_state(self) -> Dict:
        """
        Retorna o estado completo da entidade para IA/serialização.
        Este é o input que a rede neural receberá.
        """
        stats = self.stats_manager.get_stats()
        
        return {
            # Posição e movimento
            'x': self.x,
            'y': self.y,
            'vx': self.vx,
            'vy': self.vy,
            'facing_angle': self.facing_angle,
            'radius': self.radius,
            
            # Vida
            'health': self.health,
            'max_health': stats.max_health,
            'health_ratio': self.health / stats.max_health,
            
            # Status
            'is_alive': self.is_alive(),
            'is_invulnerable': self.invulnerable_time > 0,
            'is_moving': self.moving,
            
            # Arma
            'weapon': self.weapon.get_state() if self.weapon else None,
            
            # Habilidade
            'ability_cooldown': self.ability_cooldown,
            'ability_ready': self.ability_cooldown <= 0,
            
            # Classe
            'class_id': self.class_id
        }


# ============================================================================
# CLASSES IMPLEMENTADAS
# ============================================================================

@ClassRegistry.register("warrior")
class Warrior(Entity):
    """Guerreiro - classe balanceada focada em combate corpo a corpo"""
    
    display_name = "Guerreiro"
    description = "Classe equilibrada com boa defesa e dano moderado."
    default_weapon = "sword"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.ability_duration = 3.0
        self.ability_active = False
        self.ability_timer = 0
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=100,
            health_regen=1,
            speed=2.5,
            acceleration=0.5,
            friction=0.95,
            radius=30,
            mass=3.0,
            knockback_resistance=0.1,
            damage_multiplier=1.0,
            defense=2,
            armor=0.1,
            attack_speed=1.0,
            attack_range=1.0
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Fúria do Guerreiro',
            'description': 'Aumenta dano e velocidade de ataque por 3 segundos.',
            'cooldown': 10
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0 and not self.ability_active:
            from stats import StatModifier, BaseStats
            
            # Buff de dano e velocidade de ataque
            buff_stats = BaseStats(
                damage_multiplier=1.5,
                attack_speed=1.3
            )
            modifier = StatModifier("warrior_fury", buff_stats, self.ability_duration)
            self.stats_manager.add_modifier(modifier)
            
            self.ability_active = True
            self.ability_timer = self.ability_duration
            self.ability_cooldown = 10
            return True
        return False
    
    def update(self, dt: float):
        super().update(dt)
        
        if self.ability_active:
            self.ability_timer -= dt
            if self.ability_timer <= 0:
                self.ability_active = False


@ClassRegistry.register("berserker")
class Berserker(Entity):
    """Berserker - alta ofensiva, baixa defesa, fica mais forte com menos vida"""
    
    display_name = "Berserker"
    description = "Dano aumenta quanto menos vida tiver. Alta ofensiva, baixa defesa."
    default_weapon = "greatsword"
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=80,
            health_regen=0,
            speed=2.75,
            acceleration=0.6,
            friction=0.93,
            radius=32,
            mass=3.5,
            knockback_resistance=0.2,
            damage_multiplier=1.2,
            defense=0,
            armor=0,
            attack_speed=0.9,
            attack_range=1.1
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Sede de Sangue',
            'description': 'Sacrifica 20% da vida para ganhar dano massivo temporariamente.',
            'cooldown': 15
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            # Sacrificar vida
            sacrifice = self.stats_manager.get_stats().max_health * 0.2
            self.health = max(1, self.health - sacrifice)
            
            from stats import StatModifier, BaseStats
            
            # Buff de dano enorme
            buff_stats = BaseStats(
                damage_multiplier=2.0,
                attack_speed=1.5,
                speed=1.0
            )
            modifier = StatModifier("berserker_bloodlust", buff_stats, 4.0)
            self.stats_manager.add_modifier(modifier)
            
            self.ability_cooldown = 15
            return True
        return False
    
    def update(self, dt: float):
        super().update(dt)
        
        # Passive: Dano aumenta com menos vida
        stats = self.stats_manager.get_stats()
        health_ratio = self.health / stats.max_health
        
        # Remove buff antigo
        self.stats_manager.remove_modifier("berserker_passive")
        
        # Adiciona novo buff baseado na vida
        if health_ratio < 1.0:
            from stats import StatModifier, BaseStats
            bonus = (1 - health_ratio) * 0.5  # Até 50% de bônus
            passive_stats = BaseStats(damage_multiplier=1 + bonus)
            self.stats_manager.add_modifier(
                StatModifier("berserker_passive", passive_stats, -1)
            )


@ClassRegistry.register("assassin")
class Assassin(Entity):
    """Assassino - rápido, críticos altos, mas frágil"""
    
    display_name = "Assassino"
    description = "Extremamente rápido com alto dano crítico, mas pouca vida."
    default_weapon = "dagger"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.invisible = False
        self.invisible_timer = 0
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=60,
            health_regen=2,
            speed=3.5,
            acceleration=0.7,
            friction=0.92,
            radius=25,
            mass=2.0,
            knockback_resistance=0,
            damage_multiplier=1.1,
            defense=0,
            armor=0,
            attack_speed=1.5,
            attack_range=0.9
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Sombras',
            'description': 'Fica invisível por 2 segundos. Próximo ataque causa dano crítico garantido.',
            'cooldown': 12
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            self.invisible = True
            self.invisible_timer = 2.0
            
            from stats import StatModifier, BaseStats
            # Garante crítico no próximo ataque
            if self.weapon:
                self.weapon.stats.critical_chance = 1.0
            
            self.ability_cooldown = 12
            return True
        return False
    
    def update(self, dt: float):
        super().update(dt)
        
        if self.invisible:
            self.invisible_timer -= dt
            if self.invisible_timer <= 0:
                self.invisible = False
                # Restaurar chance de crítico normal
                if self.weapon:
                    self.weapon.stats.critical_chance = 0.3
    
    def draw(self, screen: pygame.Surface, against_ai: bool = False):
        """
        Desenha o Assassino.
        
        Args:
            screen: Superfície do Pygame
            against_ai: Se True, o oponente é uma IA (mostra mais transparente)
                       Se False, o oponente é um jogador (invisível total)
        """
        if self.invisible:
            if against_ai:
                # Contra IA: muito transparente (30 de alpha) - quase invisível mas visível pro jogador
                alpha = 30
            else:
                # Contra jogador: completamente invisível
                alpha = 0
            
            if alpha > 0:
                s = pygame.Surface((int(self.radius * 2), int(self.radius * 2)), pygame.SRCALPHA)
                pygame.draw.circle(s, (*self.color, alpha), 
                                 (int(self.radius), int(self.radius)), int(self.radius))
                screen.blit(s, (int(self.x - self.radius), int(self.y - self.radius)))
            # Se alpha == 0, não desenha nada (totalmente invisível)
        else:
            super().draw(screen)


@ClassRegistry.register("tank")
class Tank(Entity):
    """Tank - muito resistente, lento, focado em defesa"""
    
    display_name = "Tank"
    description = "Muito resistente com alta defesa, mas lento."
    default_weapon = "sword"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.shield_active = False
        self.shield_timer = 0
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=150,
            health_regen=3,
            speed=1.75,
            acceleration=0.3,
            friction=0.97,
            radius=38,
            mass=5.0,
            knockback_resistance=0.5,
            damage_multiplier=0.8,
            defense=5,
            armor=0.25,
            attack_speed=0.7,
            attack_range=1.0
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Escudo de Ferro',
            'description': 'Bloqueia todo dano por 2 segundos, mas não pode atacar.',
            'cooldown': 15
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            self.shield_active = True
            self.shield_timer = 2.0
            self.ability_cooldown = 15
            return True
        return False
    
    def take_damage(self, damage: float, source: 'Entity' = None, armor_penetration: float = 0) -> bool:
        if self.shield_active:
            return False  # Bloqueia todo dano
        return super().take_damage(damage, source, armor_penetration)
    
    def attack(self) -> bool:
        if self.shield_active:
            return False  # Não pode atacar com escudo ativo
        return super().attack()
    
    def update(self, dt: float):
        super().update(dt)
        
        if self.shield_active:
            self.shield_timer -= dt
            if self.shield_timer <= 0:
                self.shield_active = False
    
    def draw(self, screen: pygame.Surface):
        super().draw(screen)
        
        # Desenhar escudo se ativo
        if self.shield_active:
            pygame.draw.circle(screen, (100, 150, 255), 
                             (int(self.x), int(self.y)), 
                             int(self.radius + 10), 3)


@ClassRegistry.register("lancer")
class Lancer(Entity):
    """Lanceiro - focado em alcance e manter distância"""
    
    display_name = "Lanceiro"
    description = "Grande alcance de ataque, bom para manter distância."
    default_weapon = "spear"
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=85,
            health_regen=1.5,
            speed=2.6,
            acceleration=0.5,
            friction=0.94,
            radius=28,
            mass=2.8,
            knockback_resistance=0.15,
            damage_multiplier=1.0,
            defense=1,
            armor=0.05,
            attack_speed=1.1,
            attack_range=1.3  # 30% mais alcance
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Investida',
            'description': 'Avança rapidamente na direção que está olhando, causando dano.',
            'cooldown': 8
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            # Dash na direção que está olhando
            dash_force = 25
            self.vx += math.cos(self.facing_angle) * dash_force
            self.vy += math.sin(self.facing_angle) * dash_force
            
            # Inicia ataque automaticamente
            if self.weapon:
                self.weapon.attack()
            
            self.ability_cooldown = 8
            return True
        return False


# ============================================================================
# NOVAS CLASSES ESPECIALIZADAS PARA COMBATE EM GRUPO
# ============================================================================

@ClassRegistry.register("cleric")
class Cleric(Entity):
    """Clérigo - curandeiro principal, mantém aliados vivos"""
    
    display_name = "Clérigo"
    description = "Especialista em cura. Mantém aliados vivos com curas poderosas."
    default_weapon = "staff"
    role = "support"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.heal_target: Optional[Entity] = None
        self.passive_heal_timer = 0
        self.passive_heal_interval = 2.0  # Cura passiva a cada 2 segundos
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=75,
            health_regen=3,
            speed=2.2,
            acceleration=0.4,
            friction=0.95,
            radius=26,
            mass=2.5,
            knockback_resistance=0.1,
            damage_multiplier=0.6,  # Dano baixo
            defense=2,
            armor=0.1,
            attack_speed=0.8,
            attack_range=1.0,
            ability_power=1.5  # Cura 50% mais forte
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Cura Divina',
            'description': 'Cura todos os aliados próximos por 30% da vida máxima.',
            'cooldown': 12
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            stats = self.stats_manager.get_stats()
            heal_amount = 30 * stats.ability_power  # 30 base, escalado por ability_power
            heal_radius = 150
            
            # Curar a si mesmo
            self.heal(heal_amount)
            
            # Curar aliados próximos
            for ally in self.allies:
                if ally.is_alive():
                    dx = ally.x - self.x
                    dy = ally.y - self.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= heal_radius:
                        ally.heal(heal_amount)
                        # Aplicar heal over time também
                        ally.apply_status_effect(StatusEffect(
                            name='divine_heal_hot',
                            effect_type=StatusEffectType.HEAL_OVER_TIME,
                            duration=5.0,
                            tick_interval=1.0,
                            tick_healing=5 * stats.ability_power,
                            source=self
                        ))
            
            self.ability_cooldown = 12
            return True
        return False
    
    def update(self, dt: float):
        super().update(dt)
        
        # Cura passiva para aliados próximos
        self.passive_heal_timer += dt
        if self.passive_heal_timer >= self.passive_heal_interval:
            self.passive_heal_timer = 0
            for ally in self.allies:
                if ally.is_alive():
                    dx = ally.x - self.x
                    dy = ally.y - self.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= 80:  # Aura de cura
                        ally.heal(2)  # Cura pequena passiva
    
    def draw(self, screen: pygame.Surface):
        super().draw(screen)
        # Aura de cura
        s = pygame.Surface((160, 160), pygame.SRCALPHA)
        pygame.draw.circle(s, (100, 255, 150, 20), (80, 80), 80)
        screen.blit(s, (int(self.x - 80), int(self.y - 80)))


@ClassRegistry.register("guardian")
class Guardian(Entity):
    """Guardião - protetor de área, defende aliados com escudos"""
    
    display_name = "Guardião"
    description = "Defensor que protege aliados com escudos e absorve dano."
    default_weapon = "shield_bash"
    role = "tank"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.protection_active = False
        self.protection_timer = 0
        self.protected_allies: List[Entity] = []
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=140,
            health_regen=2,
            speed=1.8,
            acceleration=0.35,
            friction=0.96,
            radius=36,
            mass=5.5,
            knockback_resistance=0.6,
            damage_multiplier=0.7,
            defense=8,
            armor=0.35,
            attack_speed=0.6,
            attack_range=0.9
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Escudo Protetor',
            'description': 'Concede escudo a todos os aliados próximos por 4 segundos.',
            'cooldown': 18
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            shield_radius = 120
            shield_amount = 40  # Escudo de 40 HP
            
            # Aplicar escudo a si mesmo
            self.apply_status_effect(StatusEffect(
                name='guardian_shield_self',
                effect_type=StatusEffectType.SHIELD,
                duration=4.0,
                power=shield_amount * 1.5,  # Escudo maior em si mesmo
                source=self
            ))
            
            # Aplicar escudo a aliados
            for ally in self.allies:
                if ally.is_alive():
                    dx = ally.x - self.x
                    dy = ally.y - self.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= shield_radius:
                        ally.apply_status_effect(StatusEffect(
                            name='guardian_shield',
                            effect_type=StatusEffectType.SHIELD,
                            duration=4.0,
                            power=shield_amount,
                            source=self
                        ))
            
            self.ability_cooldown = 18
            return True
        return False
    
    def take_damage(self, damage: float, source: 'Entity' = None, armor_penetration: float = 0) -> bool:
        """Guardião recebe parte do dano de aliados próximos"""
        return super().take_damage(damage, source, armor_penetration)
    
    def draw(self, screen: pygame.Surface):
        super().draw(screen)
        # Aura de proteção
        s = pygame.Surface((240, 240), pygame.SRCALPHA)
        pygame.draw.circle(s, (100, 150, 255, 15), (120, 120), 120)
        pygame.draw.circle(s, (100, 150, 255, 30), (120, 120), 120, 2)
        screen.blit(s, (int(self.x - 120), int(self.y - 120)))


@ClassRegistry.register("controller")
class Controller(Entity):
    """Controlador - especialista em crowd control, stun e slow"""
    
    display_name = "Controlador"
    description = "Mestre em controle de grupo. Stuna e desacelera inimigos."
    default_weapon = "warhammer"
    role = "control"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.cc_amplifier = 1.0  # Amplificador de duração de CC
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=90,
            health_regen=1.5,
            speed=2.3,
            acceleration=0.45,
            friction=0.94,
            radius=30,
            mass=3.5,
            knockback_resistance=0.25,
            damage_multiplier=0.8,
            defense=3,
            armor=0.15,
            attack_speed=0.75,
            attack_range=1.1
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Onda de Choque',
            'description': 'Stuna todos os inimigos próximos por 1.5 segundos.',
            'cooldown': 15
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            stun_radius = 100
            stun_duration = 1.5 * self.cc_amplifier
            
            # Busca inimigos da lista global (set externamente ou via allies)
            enemies_to_check = getattr(self, '_nearby_enemies', [])
            
            # Se não tem inimigos definidos, tenta achar via referência global
            if not enemies_to_check and hasattr(self, 'game_entities'):
                enemies_to_check = [e for e in self.game_entities if e != self and e.is_alive() and (self.team == "none" or e.team != self.team)]
            
            for enemy in enemies_to_check:
                if enemy.is_alive():
                    dx = enemy.x - self.x
                    dy = enemy.y - self.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= stun_radius:
                        enemy.apply_status_effect(StatusEffect(
                            name='shockwave_stun',
                            effect_type=StatusEffectType.STUN,
                            duration=stun_duration,
                            source=self
                        ))
                        # Knockback pequeno
                        if dist > 0:
                            angle = math.atan2(dy, dx)
                            enemy.apply_knockback(15, angle)
            
            self.ability_cooldown = 15
            return True
        return False
    
    def set_enemies(self, enemies: List['Entity']):
        """Define a lista de inimigos para habilidades de CC"""
        self._nearby_enemies = enemies
    
    def draw(self, screen: pygame.Surface):
        super().draw(screen)
        # Indicador de área de CC
        if self.ability_cooldown <= 0:
            s = pygame.Surface((200, 200), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 200, 100, 20), (100, 100), 100)
            screen.blit(s, (int(self.x - 100), int(self.y - 100)))


@ClassRegistry.register("ranger")
class Ranger(Entity):
    """Arqueiro - dano à distância, mobilidade alta"""
    
    display_name = "Arqueiro"
    description = "Atacante de longa distância. Rápido e letal de longe."
    default_weapon = "bow"
    role = "ranged"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.arrows_ready = 3
        self.max_arrows = 3
        self.arrow_recharge_timer = 0
        self.arrow_recharge_time = 2.0
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=65,
            health_regen=1,
            speed=3.0,
            acceleration=0.6,
            friction=0.92,
            radius=24,
            mass=2.2,
            knockback_resistance=0.05,
            damage_multiplier=1.1,
            defense=0,
            armor=0,
            attack_speed=1.3,
            attack_range=2.0  # Dobro de alcance
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Chuva de Flechas',
            'description': 'Dispara múltiplas flechas em área, causando dano e slow.',
            'cooldown': 10
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            rain_radius = 80
            rain_distance = 150  # Distância à frente
            
            # Centro da chuva de flechas
            target_x = self.x + math.cos(self.facing_angle) * rain_distance
            target_y = self.y + math.sin(self.facing_angle) * rain_distance
            
            # Busca inimigos
            enemies_to_check = getattr(self, '_nearby_enemies', [])
            if not enemies_to_check and hasattr(self, 'game_entities'):
                enemies_to_check = [e for e in self.game_entities if e != self and e.is_alive() and (self.team == "none" or e.team != self.team)]
            
            for enemy in enemies_to_check:
                if enemy.is_alive():
                    dx = enemy.x - target_x
                    dy = enemy.y - target_y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= rain_radius:
                        # Dano
                        enemy.take_damage(20, self)
                        # Slow
                        enemy.apply_status_effect(StatusEffect(
                            name='arrow_rain_slow',
                            effect_type=StatusEffectType.SLOW,
                            duration=3.0,
                            power=0.4,  # 40% slow
                            source=self
                        ))
            
            self.ability_cooldown = 10
            return True
        return False
    
    def set_enemies(self, enemies: List['Entity']):
        """Define a lista de inimigos para habilidades"""
        self._nearby_enemies = enemies
    
    def update(self, dt: float):
        super().update(dt)
        # Recarregar flechas
        if self.arrows_ready < self.max_arrows:
            self.arrow_recharge_timer += dt
            if self.arrow_recharge_timer >= self.arrow_recharge_time:
                self.arrow_recharge_timer = 0
                self.arrows_ready += 1


@ClassRegistry.register("enchanter")
class Enchanter(Entity):
    """Encantador - buffer de aliados, aumenta poder do time"""
    
    display_name = "Encantador"
    description = "Especialista em buffs. Aumenta o poder de todo o time."
    default_weapon = "tome"
    role = "support"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.buff_active = False
        self.buff_timer = 0
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=70,
            health_regen=2,
            speed=2.4,
            acceleration=0.45,
            friction=0.94,
            radius=25,
            mass=2.3,
            knockback_resistance=0.1,
            damage_multiplier=0.7,
            defense=1,
            armor=0.05,
            attack_speed=0.9,
            attack_range=1.2,
            ability_power=1.3
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Bênção de Guerra',
            'description': 'Aumenta dano e velocidade de ataque dos aliados por 6 segundos.',
            'cooldown': 20
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0:
            buff_radius = 150
            buff_duration = 6.0
            
            # Buff a si mesmo
            self.apply_status_effect(StatusEffect(
                name='war_blessing_damage',
                effect_type=StatusEffectType.BUFF_DAMAGE,
                duration=buff_duration,
                power=0.3,  # +30% dano
                source=self
            ))
            self.apply_status_effect(StatusEffect(
                name='war_blessing_speed',
                effect_type=StatusEffectType.BUFF_SPEED,
                duration=buff_duration,
                power=0.2,  # +20% velocidade
                source=self
            ))
            
            # Buff aliados
            for ally in self.allies:
                if ally.is_alive():
                    dx = ally.x - self.x
                    dy = ally.y - self.y
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= buff_radius:
                        ally.apply_status_effect(StatusEffect(
                            name='war_blessing_damage',
                            effect_type=StatusEffectType.BUFF_DAMAGE,
                            duration=buff_duration,
                            power=0.3,
                            source=self
                        ))
                        ally.apply_status_effect(StatusEffect(
                            name='war_blessing_speed',
                            effect_type=StatusEffectType.BUFF_SPEED,
                            duration=buff_duration,
                            power=0.2,
                            source=self
                        ))
            
            self.ability_cooldown = 20
            return True
        return False
    
    def update(self, dt: float):
        super().update(dt)
        
        # Aura passiva: aliados próximos regeneram mana/cooldown mais rápido
        # (Implementado como redução de cooldown de habilidade)
        for ally in self.allies:
            if ally.is_alive() and ally.ability_cooldown > 0:
                dx = ally.x - self.x
                dy = ally.y - self.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist <= 80:
                    ally.ability_cooldown -= dt * 0.2  # 20% mais rápido
    
    def draw(self, screen: pygame.Surface):
        super().draw(screen)
        # Aura de buff
        s = pygame.Surface((160, 160), pygame.SRCALPHA)
        pygame.draw.circle(s, (255, 200, 100, 20), (80, 80), 80)
        screen.blit(s, (int(self.x - 80), int(self.y - 80)))


@ClassRegistry.register("trapper")
class Trapper(Entity):
    """Trapper - controle de terreno, coloca armadilhas"""
    
    display_name = "Trapper"
    description = "Controla o terreno com armadilhas que causam dano e CC."
    default_weapon = "trap_launcher"
    role = "control"
    
    def __init__(self, x: float, y: float, color: tuple):
        super().__init__(x, y, color)
        self.traps: List[Dict] = []
        self.max_traps = 3
        self.trap_duration = 15.0
    
    def get_default_stats(self) -> BaseStats:
        return BaseStats(
            max_health=80,
            health_regen=1.5,
            speed=2.5,
            acceleration=0.5,
            friction=0.94,
            radius=27,
            mass=2.8,
            knockback_resistance=0.15,
            damage_multiplier=0.9,
            defense=2,
            armor=0.1,
            attack_speed=0.85,
            attack_range=1.0
        )
    
    def get_ability_info(self) -> Dict:
        return {
            'name': 'Armadilha de Raiz',
            'description': 'Coloca uma armadilha que enraíza e causa dano ao ser ativada.',
            'cooldown': 8
        }
    
    def use_ability(self) -> bool:
        if self.ability_cooldown <= 0 and len(self.traps) < self.max_traps:
            # Colocar armadilha na posição à frente
            trap_distance = 50
            trap_x = self.x + math.cos(self.facing_angle) * trap_distance
            trap_y = self.y + math.sin(self.facing_angle) * trap_distance
            
            self.traps.append({
                'x': trap_x,
                'y': trap_y,
                'radius': 30,
                'duration': self.trap_duration,
                'damage': 25,
                'root_duration': 2.0,
                'active': True
            })
            
            self.ability_cooldown = 8
            return True
        return False
    
    def check_traps(self, enemies: List['Entity']):
        """Verifica se inimigos pisaram nas armadilhas"""
        for trap in self.traps[:]:  # Cópia para poder remover durante iteração
            if not trap['active']:
                continue
                
            for enemy in enemies:
                if enemy.is_alive():
                    dx = enemy.x - trap['x']
                    dy = enemy.y - trap['y']
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist <= trap['radius'] + enemy.radius:
                        # Ativar armadilha
                        enemy.take_damage(trap['damage'], self)
                        enemy.apply_status_effect(StatusEffect(
                            name='trap_root',
                            effect_type=StatusEffectType.ROOT,
                            duration=trap['root_duration'],
                            source=self
                        ))
                        trap['active'] = False
    
    def update(self, dt: float):
        super().update(dt)
        
        # Atualizar duração das armadilhas
        for trap in self.traps[:]:
            trap['duration'] -= dt
            if trap['duration'] <= 0 or not trap['active']:
                if trap in self.traps:
                    self.traps.remove(trap)
    
    def draw(self, screen: pygame.Surface):
        super().draw(screen)
        
        # Desenhar armadilhas
        for trap in self.traps:
            if trap['active']:
                # Armadilha ativa - cor marrom/amarela
                pygame.draw.circle(screen, (180, 140, 60), 
                                 (int(trap['x']), int(trap['y'])), int(trap['radius']), 2)
                # Centro da armadilha
                pygame.draw.circle(screen, (220, 180, 80), 
                                 (int(trap['x']), int(trap['y'])), 5)
            else:
                # Armadilha desativada - cinza
                pygame.draw.circle(screen, (100, 100, 100), 
                                 (int(trap['x']), int(trap['y'])), int(trap['radius']), 1)
