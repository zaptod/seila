"""
Sistema de Física
=================
Gerencia colisões e física do jogo.
"""

import math
from typing import List, TYPE_CHECKING
import pygame

if TYPE_CHECKING:
    from entities import Entity


class Physics:
    """Sistema de física para o jogo"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.restitution = 0.8  # Coeficiente de restituição (elasticidade)
    
    def handle_collisions(self, entities: List['Entity']):
        """Gerencia todas as colisões entre entidades"""
        # Colisão corpo-corpo
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                if entities[i].is_alive() and entities[j].is_alive():
                    self.check_body_collision(entities[i], entities[j])
        
        # Colisão arma-corpo e arma-arma
        for i in range(len(entities)):
            for j in range(len(entities)):
                if i != j and entities[i].is_alive() and entities[j].is_alive():
                    if hasattr(entities[i], 'weapon') and entities[i].weapon:
                        self.check_weapon_collision(entities[i], entities[j])
    
    def check_body_collision(self, entity1: 'Entity', entity2: 'Entity'):
        """Verifica e resolve colisão entre dois corpos (círculos)"""
        dx = entity2.x - entity1.x
        dy = entity2.y - entity1.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        min_distance = entity1.radius + entity2.radius
        
        if distance < min_distance and distance > 0:
            # Normalizar vetor de colisão
            nx = dx / distance
            ny = dy / distance
            
            # Calcular overlap
            overlap = min_distance - distance
            
            # Separar os círculos
            total_mass = entity1.mass + entity2.mass
            entity1.x -= nx * overlap * (entity2.mass / total_mass)
            entity1.y -= ny * overlap * (entity2.mass / total_mass)
            entity2.x += nx * overlap * (entity1.mass / total_mass)
            entity2.y += ny * overlap * (entity1.mass / total_mass)
            
            # Calcular velocidades relativas
            dvx = entity1.vx - entity2.vx
            dvy = entity1.vy - entity2.vy
            
            # Velocidade relativa na direção da colisão
            dvn = dvx * nx + dvy * ny
            
            # Não resolver se as entidades estão se afastando
            if dvn > 0:
                return
            
            # Calcular impulso
            impulse = -(1 + self.restitution) * dvn / total_mass
            
            # Aplicar impulso
            entity1.vx += impulse * entity2.mass * nx
            entity1.vy += impulse * entity2.mass * ny
            entity2.vx -= impulse * entity1.mass * nx
            entity2.vy -= impulse * entity1.mass * ny
    
    def check_weapon_collision(self, attacker: 'Entity', target: 'Entity'):
        """Verifica colisão da arma com corpo ou outra arma"""
        if not hasattr(attacker, 'weapon') or not attacker.weapon:
            return
        
        hitbox = attacker.weapon.get_hitbox()
        
        if not hitbox or not hitbox.get('active', False):
            return
        
        hitbox_type = hitbox.get('type', 'line')
        hit = False
        
        # Verificar colisão baseado no tipo de hitbox
        if hitbox_type == 'line' or hitbox_type == 'point':
            # Hitbox de linha (espadas, lanças, etc.)
            if 'start' in hitbox and 'end' in hitbox:
                hit = self.line_circle_collision(hitbox['start'], hitbox['end'], 
                                                  (target.x, target.y), target.radius)
        
        elif hitbox_type == 'circle':
            # Hitbox circular (escudo, martelo, etc.)
            if 'center' in hitbox and 'radius' in hitbox:
                hit = self.circle_circle_collision(hitbox['center'], hitbox['radius'],
                                                   (target.x, target.y), target.radius)
        
        elif hitbox_type == 'projectile':
            # Projéteis (arco, etc.)
            projectiles = hitbox.get('projectiles', [])
            for proj in projectiles:
                if proj.get('active', False):
                    proj_x, proj_y = proj.get('x', 0), proj.get('y', 0)
                    proj_radius = proj.get('radius', 5)
                    if self.circle_circle_collision((proj_x, proj_y), proj_radius,
                                                    (target.x, target.y), target.radius):
                        hit = True
                        # Desativar projétil após hit
                        if hasattr(attacker.weapon, 'on_arrow_hit'):
                            attacker.weapon.on_arrow_hit(proj)
                        break
        
        if hit:
            # Obter informações de dano da hitbox
            damage = hitbox.get('damage', attacker.weapon.stats.base_damage)
            knockback = hitbox.get('knockback', attacker.weapon.stats.knockback_force)
            armor_pen = hitbox.get('armor_penetration', 0)
            
            # Causar dano
            if target.take_damage(damage, attacker, armor_pen):
                # Knockback
                dx = target.x - attacker.x
                dy = target.y - attacker.y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance > 0:
                    angle = math.atan2(dy, dx)
                    target.apply_knockback(knockback, angle)
                
                # Callback de hit na arma
                attacker.weapon.on_hit(target, damage)
                
                # Lifesteal
                if attacker.weapon.stats.lifesteal > 0:
                    heal_amount = damage * attacker.weapon.stats.lifesteal
                    attacker.heal(heal_amount)
                
                # Aplicar efeitos especiais das armas
                # Stun (Warhammer)
                if hitbox.get('applies_stun', False) and hitbox.get('stun_effect'):
                    target.apply_status_effect(hitbox['stun_effect'])
                
                # Slow (Shield Bash)
                if hitbox.get('applies_slow', False) and hitbox.get('slow_effect'):
                    target.apply_status_effect(hitbox['slow_effect'])
        
        # Verificar colisão arma-arma (apenas para tipos de linha)
        if hitbox_type in ['line', 'point'] and 'start' in hitbox and 'end' in hitbox:
            if hasattr(target, 'weapon') and target.weapon:
                target_hitbox = target.weapon.get_hitbox()
                if target_hitbox and target_hitbox.get('active', False):
                    target_type = target_hitbox.get('type', 'line')
                    if target_type in ['line', 'point'] and 'start' in target_hitbox and 'end' in target_hitbox:
                        if self.line_line_collision(hitbox['start'], hitbox['end'],
                                                   target_hitbox['start'], target_hitbox['end']):
                            # Ricochete - ambas as armas recuam
                            self.weapon_ricochet(attacker, target)
    
    def circle_circle_collision(self, center1: tuple, radius1: float,
                                 center2: tuple, radius2: float) -> bool:
        """Verifica colisão entre dois círculos"""
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        distance = math.sqrt(dx * dx + dy * dy)
        return distance < (radius1 + radius2)
    
    def line_circle_collision(self, line_start: tuple, line_end: tuple, 
                              circle_center: tuple, radius: float) -> bool:
        """Verifica se uma linha colide com um círculo"""
        # Vetor da linha
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        
        # Vetor do início da linha até o centro do círculo
        fx = line_start[0] - circle_center[0]
        fy = line_start[1] - circle_center[1]
        
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius
        
        if a == 0:
            return False
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return False
        
        discriminant = math.sqrt(discriminant)
        
        t1 = (-b - discriminant) / (2 * a)
        t2 = (-b + discriminant) / (2 * a)
        
        # Verificar se a interseção está dentro do segmento
        if (0 <= t1 <= 1) or (0 <= t2 <= 1):
            return True
        
        # Verificar se algum ponto final está dentro do círculo
        if (fx * fx + fy * fy) <= radius * radius:
            return True
        
        ex = line_end[0] - circle_center[0]
        ey = line_end[1] - circle_center[1]
        if (ex * ex + ey * ey) <= radius * radius:
            return True
        
        return False
    
    def line_line_collision(self, p1: tuple, p2: tuple, 
                           p3: tuple, p4: tuple) -> bool:
        """Verifica se duas linhas se intersectam"""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 0.0001:
            return False
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        return 0 <= t <= 1 and 0 <= u <= 1
    
    def weapon_ricochet(self, entity1: 'Entity', entity2: 'Entity'):
        """Aplica efeito de ricochete quando duas armas colidem"""
        # Calcular direção entre os dois
        dx = entity2.x - entity1.x
        dy = entity2.y - entity1.y
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance > 0:
            nx = dx / distance
            ny = dy / distance
            
            # Força do ricochete
            ricochet_force = 10
            
            # Aplicar força oposta a cada entidade
            entity1.vx -= nx * ricochet_force
            entity1.vy -= ny * ricochet_force
            entity2.vx += nx * ricochet_force
            entity2.vy += ny * ricochet_force
    
    def constrain_to_arena(self, entity: 'Entity', arena_rect: pygame.Rect):
        """Mantém a entidade dentro da arena"""
        # Limites considerando o raio
        min_x = arena_rect.left + entity.radius
        max_x = arena_rect.right - entity.radius
        min_y = arena_rect.top + entity.radius
        max_y = arena_rect.bottom - entity.radius
        
        # Colisão com bordas
        if entity.x < min_x:
            entity.x = min_x
            entity.vx = abs(entity.vx) * self.restitution
        elif entity.x > max_x:
            entity.x = max_x
            entity.vx = -abs(entity.vx) * self.restitution
        
        if entity.y < min_y:
            entity.y = min_y
            entity.vy = abs(entity.vy) * self.restitution
        elif entity.y > max_y:
            entity.y = max_y
            entity.vy = -abs(entity.vy) * self.restitution
    
    def constrain_projectile_to_arena(self, projectile: dict, arena_rect: pygame.Rect, 
                                       radius: float = 5) -> bool:
        """
        Verifica e trata colisão de projétil com bordas da arena.
        Retorna True se o projétil deve ser desativado.
        """
        x = projectile.get('x', 0)
        y = projectile.get('y', 0)
        
        # Verificar se está fora da arena
        if (x - radius < arena_rect.left or 
            x + radius > arena_rect.right or
            y - radius < arena_rect.top or 
            y + radius > arena_rect.bottom):
            return True  # Desativar projétil
        
        return False
    
    def check_projectiles_arena_collision(self, entities: List['Entity'], arena_rect: pygame.Rect):
        """Verifica colisão de todos os projéteis com as bordas da arena"""
        for entity in entities:
            if not entity.is_alive() or not hasattr(entity, 'weapon') or not entity.weapon:
                continue
            
            weapon = entity.weapon
            
            # Arco - flechas
            if hasattr(weapon, 'arrows'):
                for arrow in weapon.arrows[:]:
                    if arrow.get('active', False):
                        if self.constrain_projectile_to_arena(arrow, arena_rect, radius=3):
                            arrow['active'] = False
            
            # Trap Launcher - armadilhas em voo
            if hasattr(weapon, 'launched_traps'):
                for trap in weapon.launched_traps[:]:
                    if trap.get('in_flight', False):
                        if self.constrain_projectile_to_arena(trap, arena_rect, radius=8):
                            # Armadilha colide com borda e cai ali
                            trap['in_flight'] = False
                            # Converter para armadilha no chão na posição atual
                            if hasattr(weapon, 'ground_traps') and len(weapon.ground_traps) < weapon.max_ground_traps:
                                # Clamp position to arena
                                trap_x = max(arena_rect.left + 20, min(trap['x'], arena_rect.right - 20))
                                trap_y = max(arena_rect.top + 20, min(trap['y'], arena_rect.bottom - 20))
                                weapon.ground_traps.append({
                                    'x': trap_x,
                                    'y': trap_y,
                                    'radius': weapon.stats.width,
                                    'duration': weapon.trap_duration,
                                    'damage': weapon.stats.special_effects['trap_damage'],
                                    'root_duration': weapon.stats.special_effects['root_duration'],
                                    'active': True,
                                    'armed_time': 0.3
                                })
                            weapon.launched_traps.remove(trap)
