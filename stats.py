"""
Sistema de Atributos e Stats
============================
Sistema flexível para definir atributos de entidades, armas e habilidades.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, TYPE_CHECKING
import copy

if TYPE_CHECKING:
    from entities import Entity


@dataclass
class BaseStats:
    """Atributos base de uma entidade"""
    max_health: float = 100
    health_regen: float = 0  # Regeneração por segundo
    
    # Movimento
    speed: float = 2.5
    acceleration: float = 0.5
    friction: float = 0.95
    
    # Físico
    radius: float = 30
    mass: float = 3.0
    knockback_resistance: float = 0  # 0-1, reduz knockback recebido
    
    # Combate
    damage_multiplier: float = 1.0  # Multiplicador de dano causado
    defense: float = 0  # Redução de dano fixo
    armor: float = 0  # Redução de dano percentual (0-1)
    
    # Arma
    attack_speed: float = 1.0  # Multiplicador de velocidade de ataque
    attack_range: float = 1.0  # Multiplicador de alcance
    
    # Habilidade
    ability_cooldown_reduction: float = 0  # 0-1, redução de cooldown
    ability_power: float = 1.0  # Multiplicador de poder de habilidade
    
    def combine(self, other: 'BaseStats') -> 'BaseStats':
        """Combina dois stats (para buffs/equipamentos)"""
        result = copy.deepcopy(self)
        result.max_health += other.max_health
        result.health_regen += other.health_regen
        result.speed += other.speed
        result.acceleration += other.acceleration
        result.mass += other.mass
        result.knockback_resistance = min(1, result.knockback_resistance + other.knockback_resistance)
        result.damage_multiplier *= other.damage_multiplier
        result.defense += other.defense
        result.armor = min(0.9, result.armor + other.armor)  # Cap de 90% armor
        result.attack_speed *= other.attack_speed
        result.attack_range *= other.attack_range
        result.ability_cooldown_reduction = min(0.8, result.ability_cooldown_reduction + other.ability_cooldown_reduction)
        result.ability_power *= other.ability_power
        return result


@dataclass
class WeaponStats:
    """Atributos específicos de armas"""
    base_damage: float = 10
    attack_cooldown: float = 0.5  # Segundos entre ataques
    attack_duration: float = 0.3  # Duração da animação de ataque
    
    # Alcance e tamanho
    range: float = 50  # Alcance base
    width: float = 8   # Largura/área
    
    # Knockback
    knockback_force: float = 15
    
    # Especiais
    armor_penetration: float = 0  # 0-1, ignora armor
    lifesteal: float = 0  # 0-1, cura baseada no dano
    critical_chance: float = 0  # 0-1
    critical_multiplier: float = 2.0
    
    # Efeitos especiais (dict para flexibilidade)
    special_effects: Dict = field(default_factory=dict)


@dataclass  
class AbilityStats:
    """Atributos de habilidades"""
    cooldown: float = 10  # Segundos
    duration: float = 0  # Duração do efeito (0 = instantâneo)
    damage: float = 0
    healing: float = 0
    range: float = 100
    radius: float = 0  # Para habilidades em área
    
    # Custos
    health_cost: float = 0  # Custo em vida
    
    # Efeitos especiais
    special_effects: Dict = field(default_factory=dict)


# ============================================================================
# SISTEMA DE STATUS EFFECTS (Efeitos de Status)
# ============================================================================

@dataclass
class StatusEffect:
    """
    Efeito de status que pode ser aplicado a entidades.
    Usado para controle de grupo, debuffs, buffs, etc.
    """
    name: str  # Nome único do efeito
    effect_type: str  # Tipo: 'stun', 'slow', 'root', 'silence', 'burn', 'heal_over_time', 'shield', 'buff', 'debuff'
    duration: float  # Duração em segundos
    power: float = 1.0  # Intensidade do efeito (ex: 0.5 = 50% slow)
    tick_interval: float = 0  # Intervalo para efeitos que dão tick (0 = sem tick)
    tick_damage: float = 0  # Dano por tick
    tick_healing: float = 0  # Cura por tick
    source: Optional['Entity'] = None  # Quem aplicou o efeito
    stacks: int = 1  # Quantidade de stacks
    max_stacks: int = 1  # Máximo de stacks
    
    # Estado interno
    time_remaining: float = field(init=False)
    tick_timer: float = field(init=False)
    
    def __post_init__(self):
        self.time_remaining = self.duration
        self.tick_timer = self.tick_interval


class StatusEffectType:
    """Constantes para tipos de efeitos de status"""
    # Controle
    STUN = 'stun'           # Não pode se mover nem atacar
    SLOW = 'slow'           # Movimento reduzido
    ROOT = 'root'           # Não pode se mover, mas pode atacar
    SILENCE = 'silence'     # Não pode usar habilidade
    BLIND = 'blind'         # Ataques têm chance de errar
    TAUNT = 'taunt'         # Forçado a atacar quem aplicou
    FEAR = 'fear'           # Foge do aplicador
    KNOCKUP = 'knockup'     # Lançado no ar (stunned + deslocamento)
    
    # Dano ao longo do tempo
    BURN = 'burn'           # Dano de fogo
    POISON = 'poison'       # Dano de veneno
    BLEED = 'bleed'         # Dano de sangramento
    
    # Cura e proteção
    HEAL_OVER_TIME = 'heal_over_time'  # Cura ao longo do tempo
    SHIELD = 'shield'       # Escudo que absorve dano
    INVULNERABLE = 'invulnerable'  # Invulnerável a dano
    
    # Buffs/Debuffs
    BUFF_DAMAGE = 'buff_damage'       # Aumenta dano
    BUFF_SPEED = 'buff_speed'         # Aumenta velocidade
    BUFF_DEFENSE = 'buff_defense'     # Aumenta defesa
    DEBUFF_DAMAGE = 'debuff_damage'   # Reduz dano
    DEBUFF_DEFENSE = 'debuff_defense' # Reduz defesa
    
    # Especiais
    MARKED = 'marked'       # Marcado - recebe mais dano
    REVEALED = 'revealed'   # Revelado - não pode ficar invisível


class StatusEffectManager:
    """
    Gerencia todos os status effects de uma entidade.
    """
    
    def __init__(self, owner: 'Entity'):
        self.owner = owner
        self.effects: Dict[str, StatusEffect] = {}
        self.shield_amount: float = 0  # Escudo atual
    
    def apply_effect(self, effect: StatusEffect) -> bool:
        """
        Aplica um efeito de status.
        Retorna True se foi aplicado, False se foi resistido/imune.
        """
        # Verificar imunidade (ex: Tank com escudo ativo)
        if hasattr(self.owner, 'is_immune_to_cc') and self.owner.is_immune_to_cc():
            if effect.effect_type in [StatusEffectType.STUN, StatusEffectType.SLOW, 
                                       StatusEffectType.ROOT, StatusEffectType.FEAR]:
                return False
        
        # Se já tem o efeito, verificar stacking
        if effect.name in self.effects:
            existing = self.effects[effect.name]
            if existing.stacks < existing.max_stacks:
                existing.stacks += 1
                existing.power = min(existing.power * 1.5, 0.9)  # Cap em 90%
            # Renovar duração
            existing.time_remaining = max(existing.time_remaining, effect.duration)
            return True
        
        # Aplicar novo efeito
        self.effects[effect.name] = effect
        
        # Efeitos especiais na aplicação
        if effect.effect_type == StatusEffectType.SHIELD:
            self.shield_amount += effect.power
        
        return True
    
    def remove_effect(self, name: str):
        """Remove um efeito pelo nome"""
        if name in self.effects:
            effect = self.effects[name]
            if effect.effect_type == StatusEffectType.SHIELD:
                self.shield_amount = max(0, self.shield_amount - effect.power)
            del self.effects[name]
    
    def has_effect(self, effect_type: str) -> bool:
        """Verifica se tem um tipo de efeito ativo"""
        return any(e.effect_type == effect_type for e in self.effects.values())
    
    def get_effect(self, name: str) -> Optional[StatusEffect]:
        """Retorna um efeito pelo nome"""
        return self.effects.get(name)
    
    def is_stunned(self) -> bool:
        """Verifica se está stunado"""
        return self.has_effect(StatusEffectType.STUN) or self.has_effect(StatusEffectType.KNOCKUP)
    
    def is_rooted(self) -> bool:
        """Verifica se está enraizado"""
        return self.has_effect(StatusEffectType.ROOT) or self.is_stunned()
    
    def is_silenced(self) -> bool:
        """Verifica se está silenciado"""
        return self.has_effect(StatusEffectType.SILENCE) or self.is_stunned()
    
    def is_slowed(self) -> bool:
        """Verifica se está com slow"""
        return self.has_effect(StatusEffectType.SLOW)
    
    def get_slow_factor(self) -> float:
        """Retorna o fator de slow total (1.0 = sem slow)"""
        slow_factor = 1.0
        for effect in self.effects.values():
            if effect.effect_type == StatusEffectType.SLOW:
                slow_factor *= (1 - effect.power)
        return max(0.1, slow_factor)  # Mínimo 10% de velocidade
    
    def get_speed_multiplier(self) -> float:
        """Retorna multiplicador de velocidade baseado em buffs/debuffs e slow"""
        multiplier = self.get_slow_factor()
        for effect in self.effects.values():
            if effect.effect_type == StatusEffectType.BUFF_SPEED:
                multiplier *= (1 + effect.power)
        return max(0.1, multiplier)  # Mínimo 10% de velocidade
    
    def get_damage_multiplier(self) -> float:
        """Retorna multiplicador de dano baseado em buffs/debuffs"""
        multiplier = 1.0
        for effect in self.effects.values():
            if effect.effect_type == StatusEffectType.BUFF_DAMAGE:
                multiplier *= (1 + effect.power)
            elif effect.effect_type == StatusEffectType.DEBUFF_DAMAGE:
                multiplier *= (1 - effect.power)
            elif effect.effect_type == StatusEffectType.MARKED:
                multiplier *= (1 + effect.power * 0.5)  # Marked = +50% dano recebido
        return max(0.1, multiplier)
    
    def get_shield(self) -> float:
        """Retorna a quantidade de escudo atual"""
        return self.shield_amount
    
    def absorb_damage(self, damage: float) -> float:
        """
        Tenta absorver dano com escudo.
        Retorna o dano restante após absorção.
        """
        if self.shield_amount <= 0:
            return damage
        
        absorbed = min(self.shield_amount, damage)
        self.shield_amount -= absorbed
        
        # Remover efeitos de shield se escudo acabou
        if self.shield_amount <= 0:
            to_remove = [name for name, e in self.effects.items() 
                        if e.effect_type == StatusEffectType.SHIELD]
            for name in to_remove:
                del self.effects[name]
        
        return damage - absorbed
    
    def update(self, dt: float) -> List[Dict]:
        """
        Atualiza todos os efeitos.
        Retorna lista de eventos (dano/cura aplicados).
        """
        events = []
        expired = []
        
        for name, effect in self.effects.items():
            # Atualizar duração
            effect.time_remaining -= dt
            
            # Processar ticks
            if effect.tick_interval > 0:
                effect.tick_timer -= dt
                if effect.tick_timer <= 0:
                    effect.tick_timer = effect.tick_interval
                    
                    # Aplicar dano de tick
                    if effect.tick_damage > 0:
                        actual_damage = effect.tick_damage * effect.stacks
                        self.owner.take_damage(actual_damage, effect.source, armor_penetration=0.5)
                        events.append({
                            'type': 'tick_damage',
                            'amount': actual_damage,
                            'effect': effect.effect_type
                        })
                    
                    # Aplicar cura de tick
                    if effect.tick_healing > 0:
                        actual_healing = effect.tick_healing * effect.stacks
                        self.owner.heal(actual_healing)
                        events.append({
                            'type': 'tick_healing',
                            'amount': actual_healing,
                            'effect': effect.effect_type
                        })
            
            # Marcar expirados
            if effect.time_remaining <= 0:
                expired.append(name)
        
        # Remover expirados
        for name in expired:
            self.remove_effect(name)
            events.append({'type': 'effect_expired', 'name': name})
        
        return events
    
    def clear_all(self):
        """Remove todos os efeitos"""
        self.effects.clear()
        self.shield_amount = 0
    
    def get_active_effects_info(self) -> List[Dict]:
        """Retorna informações sobre efeitos ativos (para UI/IA)"""
        return [
            {
                'name': e.name,
                'type': e.effect_type,
                'duration': e.time_remaining,
                'power': e.power,
                'stacks': e.stacks
            }
            for e in self.effects.values()
        ]


class StatModifier:
    """Modificador temporário de stats (buffs/debuffs)"""
    
    def __init__(self, name: str, stats: BaseStats, duration: float = -1):
        """
        Args:
            name: Nome do modificador
            stats: Stats a adicionar/multiplicar
            duration: Duração em segundos (-1 = permanente)
        """
        self.name = name
        self.stats = stats
        self.duration = duration
        self.time_remaining = duration
    
    def update(self, dt: float) -> bool:
        """Atualiza o timer. Retorna False se expirou."""
        if self.duration < 0:
            return True
        self.time_remaining -= dt
        return self.time_remaining > 0


class StatsManager:
    """Gerencia stats base + modificadores de uma entidade"""
    
    def __init__(self, base_stats: BaseStats):
        self.base_stats = base_stats
        self.modifiers: list[StatModifier] = []
        self._cached_stats: Optional[BaseStats] = None
        self._cache_dirty = True
    
    def add_modifier(self, modifier: StatModifier):
        """Adiciona um modificador de stats"""
        self.modifiers.append(modifier)
        self._cache_dirty = True
    
    def remove_modifier(self, name: str):
        """Remove modificador pelo nome"""
        self.modifiers = [m for m in self.modifiers if m.name != name]
        self._cache_dirty = True
    
    def update(self, dt: float):
        """Atualiza modificadores temporários"""
        expired = []
        for mod in self.modifiers:
            if not mod.update(dt):
                expired.append(mod)
        
        if expired:
            for mod in expired:
                self.modifiers.remove(mod)
            self._cache_dirty = True
    
    def get_stats(self) -> BaseStats:
        """Retorna stats combinados (base + modificadores)"""
        if self._cache_dirty:
            self._cached_stats = copy.deepcopy(self.base_stats)
            for mod in self.modifiers:
                self._cached_stats = self._cached_stats.combine(mod.stats)
            self._cache_dirty = False
        return self._cached_stats
    
    def calculate_damage(self, base_damage: float, weapon_stats: WeaponStats) -> float:
        """Calcula dano final baseado nos stats"""
        import random
        
        stats = self.get_stats()
        damage = base_damage * stats.damage_multiplier
        
        # Critical hit
        if random.random() < weapon_stats.critical_chance:
            damage *= weapon_stats.critical_multiplier
        
        return damage
    
    def calculate_damage_taken(self, incoming_damage: float, armor_penetration: float = 0) -> float:
        """Calcula dano recebido após defesas"""
        stats = self.get_stats()
        
        # Aplicar defesa fixa
        damage = max(0, incoming_damage - stats.defense)
        
        # Aplicar armor (reduzido pela penetração)
        effective_armor = stats.armor * (1 - armor_penetration)
        damage = damage * (1 - effective_armor)
        
        return damage
    
    def calculate_knockback(self, knockback_force: float) -> float:
        """Calcula knockback recebido"""
        stats = self.get_stats()
        return knockback_force * (1 - stats.knockback_resistance)
