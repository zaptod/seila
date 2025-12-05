"""
Configurações de Balanceamento do Jogo
======================================

Este arquivo centraliza todos os valores de balanceamento para facilitar ajustes.
Importado por entities.py e weapons.py.

NOTAS DE BALANCEAMENTO (v2.1):
- DPS esperado por categoria:
  - Tank: ~15-20 DPS
  - DPS Melee: ~25-35 DPS
  - DPS Ranged: ~20-30 DPS
  - Support: ~10-15 DPS

- Tempo médio para matar (TTK) esperado:
  - Tank vs Tank: 15-20s
  - DPS vs DPS: 5-8s
  - DPS vs Tank: 10-15s
  - DPS vs Support: 3-5s

- Cooldowns de habilidades:
  - Utilitário: 8-12s
  - Ofensivo forte: 12-18s
  - Ultimate/Defensivo: 15-25s
"""

from dataclasses import dataclass
from typing import Dict

# =============================================================================
# CONFIGURAÇÕES DE CLASSES
# =============================================================================

CLASS_STATS = {
    # CLASSES ORIGINAIS
    "warrior": {
        "max_health": 100,
        "health_regen": 1.5,
        "speed": 2.5,
        "acceleration": 0.5,
        "friction": 0.95,
        "radius": 30,
        "mass": 3.0,
        "knockback_resistance": 0.15,
        "damage_multiplier": 1.0,
        "defense": 3,
        "armor": 0.12,
        "attack_speed": 1.0,
        "attack_range": 1.0,
        "ability_cooldown": 10,
    },
    
    "berserker": {
        "max_health": 90,
        "health_regen": 0.5,
        "speed": 2.8,
        "acceleration": 0.6,
        "friction": 0.93,
        "radius": 32,
        "mass": 3.5,
        "knockback_resistance": 0.2,
        "damage_multiplier": 1.15,
        "defense": 1,
        "armor": 0.05,
        "attack_speed": 0.95,
        "attack_range": 1.1,
        "ability_cooldown": 14,
    },
    
    "assassin": {
        "max_health": 70,
        "health_regen": 2.0,
        "speed": 3.3,
        "acceleration": 0.7,
        "friction": 0.92,
        "radius": 25,
        "mass": 2.0,
        "knockback_resistance": 0.0,
        "damage_multiplier": 1.05,
        "defense": 0,
        "armor": 0.0,
        "attack_speed": 1.4,
        "attack_range": 0.85,
        "ability_cooldown": 12,
    },
    
    "tank": {
        "max_health": 160,
        "health_regen": 2.5,
        "speed": 1.8,
        "acceleration": 0.35,
        "friction": 0.97,
        "radius": 38,
        "mass": 5.5,
        "knockback_resistance": 0.5,
        "damage_multiplier": 0.75,
        "defense": 6,
        "armor": 0.28,
        "attack_speed": 0.7,
        "attack_range": 1.0,
        "ability_cooldown": 14,
    },
    
    "lancer": {
        "max_health": 90,
        "health_regen": 1.5,
        "speed": 2.6,
        "acceleration": 0.5,
        "friction": 0.94,
        "radius": 28,
        "mass": 2.8,
        "knockback_resistance": 0.15,
        "damage_multiplier": 0.95,
        "defense": 2,
        "armor": 0.08,
        "attack_speed": 1.1,
        "attack_range": 1.35,
        "ability_cooldown": 8,
    },
    
    # NOVAS CLASSES
    "cleric": {
        "max_health": 85,
        "health_regen": 3.0,
        "speed": 2.2,
        "acceleration": 0.4,
        "friction": 0.95,
        "radius": 26,
        "mass": 2.5,
        "knockback_resistance": 0.1,
        "damage_multiplier": 0.55,
        "defense": 2,
        "armor": 0.1,
        "attack_speed": 0.85,
        "attack_range": 1.0,
        "ability_power": 1.4,
        "ability_cooldown": 10,
        "heal_amount": 35,
        "heal_radius": 150,
    },
    
    "guardian": {
        "max_health": 150,
        "health_regen": 2.0,
        "speed": 1.85,
        "acceleration": 0.35,
        "friction": 0.96,
        "radius": 36,
        "mass": 5.5,
        "knockback_resistance": 0.55,
        "damage_multiplier": 0.65,
        "defense": 7,
        "armor": 0.32,
        "attack_speed": 0.65,
        "attack_range": 0.9,
        "ability_cooldown": 16,
        "shield_amount": 45,
        "shield_duration": 4.0,
    },
    
    "controller": {
        "max_health": 75,
        "health_regen": 1.5,
        "speed": 2.3,
        "acceleration": 0.45,
        "friction": 0.95,
        "radius": 27,
        "mass": 2.5,
        "knockback_resistance": 0.05,
        "damage_multiplier": 0.7,
        "defense": 1,
        "armor": 0.05,
        "attack_speed": 0.9,
        "attack_range": 1.1,
        "ability_power": 1.3,
        "ability_cooldown": 12,
        "slow_power": 0.45,
        "slow_duration": 3.5,
    },
    
    "ranger": {
        "max_health": 80,
        "health_regen": 1.5,
        "speed": 2.7,
        "acceleration": 0.55,
        "friction": 0.94,
        "radius": 26,
        "mass": 2.3,
        "knockback_resistance": 0.05,
        "damage_multiplier": 1.1,
        "defense": 1,
        "armor": 0.05,
        "attack_speed": 1.0,
        "attack_range": 1.0,
        "ability_cooldown": 14,
        "arrow_rain_damage": 12,
        "arrow_rain_radius": 100,
    },
    
    "enchanter": {
        "max_health": 70,
        "health_regen": 1.5,
        "speed": 2.4,
        "acceleration": 0.45,
        "friction": 0.95,
        "radius": 26,
        "mass": 2.3,
        "knockback_resistance": 0.0,
        "damage_multiplier": 0.6,
        "defense": 0,
        "armor": 0.0,
        "attack_speed": 0.85,
        "attack_range": 1.0,
        "ability_power": 1.3,
        "ability_cooldown": 18,
        "buff_damage": 0.25,
        "buff_speed": 0.2,
        "buff_duration": 6.0,
    },
    
    "trapper": {
        "max_health": 80,
        "health_regen": 1.5,
        "speed": 2.5,
        "acceleration": 0.5,
        "friction": 0.94,
        "radius": 27,
        "mass": 2.5,
        "knockback_resistance": 0.1,
        "damage_multiplier": 0.85,
        "defense": 2,
        "armor": 0.08,
        "attack_speed": 0.9,
        "attack_range": 1.0,
        "ability_cooldown": 10,
        "trap_damage": 25,
        "trap_root_duration": 2.0,
        "max_traps": 3,
    },
}

# =============================================================================
# CONFIGURAÇÕES DE ARMAS
# =============================================================================

WEAPON_STATS = {
    "sword": {
        "base_damage": 15,
        "attack_cooldown": 0.6,
        "attack_duration": 0.35,
        "range": 55,
        "width": 8,
        "knockback_force": 8,
        "critical_chance": 0.12,
        "critical_multiplier": 1.8,
        "armor_penetration": 0.0,
    },
    
    "greatsword": {
        "base_damage": 28,
        "attack_cooldown": 1.1,
        "attack_duration": 0.55,
        "range": 70,
        "width": 15,
        "knockback_force": 18,
        "critical_chance": 0.08,
        "critical_multiplier": 2.0,
        "armor_penetration": 0.15,
    },
    
    "dagger": {
        "base_damage": 10,
        "attack_cooldown": 0.35,
        "attack_duration": 0.18,
        "range": 35,
        "width": 5,
        "knockback_force": 3,
        "critical_chance": 0.28,
        "critical_multiplier": 2.2,
        "armor_penetration": 0.1,
    },
    
    "spear": {
        "base_damage": 14,
        "attack_cooldown": 0.7,
        "attack_duration": 0.3,
        "range": 85,
        "width": 6,
        "knockback_force": 10,
        "critical_chance": 0.1,
        "critical_multiplier": 1.7,
        "armor_penetration": 0.2,
    },
    
    "staff": {
        "base_damage": 8,
        "attack_cooldown": 0.8,
        "attack_duration": 0.4,
        "range": 100,
        "width": 5,
        "knockback_force": 5,
        "critical_chance": 0.05,
        "critical_multiplier": 1.5,
        "armor_penetration": 0.0,
        "heal_amount": 12,
        "heal_range": 100,
    },
    
    "bow": {
        "base_damage": 16,
        "attack_cooldown": 1.0,
        "attack_duration": 0.25,
        "range": 220,
        "width": 3,
        "knockback_force": 6,
        "critical_chance": 0.18,
        "critical_multiplier": 2.3,
        "armor_penetration": 0.1,
        "arrow_speed": 14,
    },
    
    "warhammer": {
        "base_damage": 22,
        "attack_cooldown": 1.4,
        "attack_duration": 0.7,
        "range": 50,
        "width": 22,
        "knockback_force": 22,
        "critical_chance": 0.1,
        "critical_multiplier": 1.8,
        "armor_penetration": 0.25,
        "stun_duration": 0.7,
    },
    
    "tome": {
        "base_damage": 10,
        "attack_cooldown": 0.9,
        "attack_duration": 0.4,
        "range": 90,
        "width": 8,
        "knockback_force": 4,
        "critical_chance": 0.08,
        "critical_multiplier": 1.5,
        "armor_penetration": 0.15,
        "buff_damage": 0.12,
        "buff_duration": 4.0,
    },
    
    "shield_bash": {
        "base_damage": 10,
        "attack_cooldown": 0.9,
        "attack_duration": 0.35,
        "range": 40,
        "width": 28,
        "knockback_force": 20,
        "critical_chance": 0.05,
        "critical_multiplier": 1.5,
        "armor_penetration": 0.0,
        "slow_duration": 1.2,
        "slow_power": 0.35,
    },
    
    "trap_launcher": {
        "base_damage": 18,
        "attack_cooldown": 1.5,
        "attack_duration": 0.4,
        "range": 180,
        "width": 15,
        "knockback_force": 5,
        "critical_chance": 0.1,
        "critical_multiplier": 1.6,
        "armor_penetration": 0.0,
        "trap_damage": 20,
        "root_duration": 1.8,
        "projectile_speed": 10,
        "max_traps": 4,
        "trap_duration": 12,
    },
}

# =============================================================================
# MULTIPLICADORES GLOBAIS (para ajustes rápidos)
# =============================================================================

GLOBAL_BALANCE = {
    "damage_multiplier": 1.0,       # Multiplicador global de dano
    "healing_multiplier": 1.0,      # Multiplicador global de cura
    "cooldown_multiplier": 1.0,     # Multiplicador global de cooldowns
    "movement_speed_multiplier": 1.0,  # Multiplicador global de velocidade
    "cc_duration_multiplier": 1.0,  # Multiplicador de duração de CC
}

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def get_class_stats(class_name: str) -> dict:
    """Retorna as stats de uma classe com multiplicadores globais aplicados"""
    stats = CLASS_STATS.get(class_name.lower(), CLASS_STATS["warrior"]).copy()
    stats["damage_multiplier"] *= GLOBAL_BALANCE["damage_multiplier"]
    stats["speed"] *= GLOBAL_BALANCE["movement_speed_multiplier"]
    if "ability_cooldown" in stats:
        stats["ability_cooldown"] *= GLOBAL_BALANCE["cooldown_multiplier"]
    return stats

def get_weapon_stats(weapon_name: str) -> dict:
    """Retorna as stats de uma arma com multiplicadores globais aplicados"""
    stats = WEAPON_STATS.get(weapon_name.lower(), WEAPON_STATS["sword"]).copy()
    stats["base_damage"] *= GLOBAL_BALANCE["damage_multiplier"]
    stats["attack_cooldown"] *= GLOBAL_BALANCE["cooldown_multiplier"]
    if "heal_amount" in stats:
        stats["heal_amount"] *= GLOBAL_BALANCE["healing_multiplier"]
    if "stun_duration" in stats:
        stats["stun_duration"] *= GLOBAL_BALANCE["cc_duration_multiplier"]
    if "slow_duration" in stats:
        stats["slow_duration"] *= GLOBAL_BALANCE["cc_duration_multiplier"]
    if "root_duration" in stats:
        stats["root_duration"] *= GLOBAL_BALANCE["cc_duration_multiplier"]
    return stats


# =============================================================================
# ANÁLISE DE BALANCEAMENTO (para debug)
# =============================================================================

def print_balance_analysis():
    """Imprime análise de balanceamento para debug"""
    print("=" * 60)
    print("ANÁLISE DE BALANCEAMENTO")
    print("=" * 60)
    
    print("\n📊 CLASSES - DPS Teórico (dano base / cooldown):")
    print("-" * 40)
    
    for class_name, stats in CLASS_STATS.items():
        # DPS estimado com arma padrão
        default_weapon = {
            "warrior": "sword", "berserker": "greatsword", "assassin": "dagger",
            "tank": "sword", "lancer": "spear", "cleric": "staff",
            "guardian": "shield_bash", "controller": "staff", "ranger": "bow",
            "enchanter": "tome", "trapper": "trap_launcher"
        }.get(class_name, "sword")
        
        weapon = WEAPON_STATS.get(default_weapon, WEAPON_STATS["sword"])
        dps = (weapon["base_damage"] * stats["damage_multiplier"]) / weapon["attack_cooldown"]
        ehp = stats["max_health"] / (1 - stats["armor"])  # Effective HP
        
        print(f"  {class_name.capitalize():12} HP:{stats['max_health']:3} EHP:{ehp:5.0f} DPS:{dps:5.1f}")
    
    print("\n⚔️ ARMAS - DPS Base:")
    print("-" * 40)
    
    for weapon_name, stats in WEAPON_STATS.items():
        dps = stats["base_damage"] / stats["attack_cooldown"]
        crit_dps = dps * (1 + stats["critical_chance"] * (stats["critical_multiplier"] - 1))
        print(f"  {weapon_name.capitalize():14} Dano:{stats['base_damage']:2} CD:{stats['attack_cooldown']:.1f}s DPS:{dps:5.1f} CritDPS:{crit_dps:5.1f}")


if __name__ == "__main__":
    print_balance_analysis()
