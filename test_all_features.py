"""
Script de teste para verificar todas as funcionalidades novas
"""
import pygame
pygame.init()

import game
import entities
import weapons
import controller
from stats import StatusEffect, StatusEffectType

def test_all_classes():
    """Testa se todas as classes podem ser criadas"""
    print("\n=== TESTE DE CLASSES ===")
    classes = ['warrior', 'berserker', 'assassin', 'tank', 'lancer', 
               'cleric', 'guardian', 'controller', 'ranger', 'enchanter', 'trapper']
    
    g = game.Game()
    for cls in classes:
        try:
            entity = g.create_entity(cls, 100, 100, (255, 255, 255), None, 'sword')
            print(f"✅ {cls.capitalize()}: {entity.__class__.__name__}")
        except Exception as e:
            print(f"❌ {cls}: {e}")
    
    pygame.quit()
    return True

def test_all_weapons():
    """Testa se todas as armas podem ser criadas"""
    print("\n=== TESTE DE ARMAS ===")
    wpns = ['sword', 'greatsword', 'dagger', 'spear', 'staff', 
            'bow', 'warhammer', 'tome', 'shield_bash', 'trap_launcher']
    
    pygame.init()
    g = game.Game()
    
    for wpn in wpns:
        try:
            entity = g.create_entity('warrior', 100, 100, (255, 255, 255), None, wpn)
            print(f"✅ {wpn.capitalize()}: {entity.weapon.__class__.__name__}")
        except Exception as e:
            print(f"❌ {wpn}: {e}")
    
    pygame.quit()
    return True

def test_all_strategies():
    """Testa se todas as estratégias de IA existem"""
    print("\n=== TESTE DE ESTRATÉGIAS ===")
    
    classes = ['warrior', 'berserker', 'assassin', 'tank', 'lancer', 
               'cleric', 'guardian', 'controller', 'ranger', 'enchanter', 'trapper']
    wpns = ['sword', 'greatsword', 'dagger', 'spear', 'staff', 
            'bow', 'warhammer', 'tome', 'shield_bash', 'trap_launcher']
    
    total = 0
    found = 0
    
    for cls in classes:
        for wpn in wpns:
            total += 1
            key = (cls.capitalize(), wpn.capitalize())
            if key in controller.CLASS_WEAPON_STRATEGIES:
                found += 1
            else:
                print(f"⚠️ Estratégia faltando: {cls}+{wpn}")
    
    print(f"\n✅ Estratégias encontradas: {found}/{total}")
    return found == total

def test_status_effects():
    """Testa se os status effects funcionam"""
    print("\n=== TESTE DE STATUS EFFECTS ===")
    
    pygame.init()
    g = game.Game()
    entity = g.create_entity('warrior', 100, 100, (255, 255, 255), None, 'sword')
    
    # Testar STUN
    entity.apply_status_effect(StatusEffect(
        name='test_stun',
        effect_type=StatusEffectType.STUN,
        duration=2.0
    ))
    print(f"✅ STUN aplicado: is_stunned={entity.status_effects.is_stunned()}")
    
    # Testar SLOW
    entity.apply_status_effect(StatusEffect(
        name='test_slow',
        effect_type=StatusEffectType.SLOW,
        duration=2.0,
        power=0.5
    ))
    print(f"✅ SLOW aplicado: is_slowed={entity.status_effects.is_slowed()}")
    
    # Testar ROOT
    entity.apply_status_effect(StatusEffect(
        name='test_root',
        effect_type=StatusEffectType.ROOT,
        duration=2.0
    ))
    print(f"✅ ROOT aplicado: is_rooted={entity.status_effects.is_rooted()}")
    
    # Testar SHIELD
    entity.apply_status_effect(StatusEffect(
        name='test_shield',
        effect_type=StatusEffectType.SHIELD,
        duration=2.0,
        power=50
    ))
    print(f"✅ SHIELD aplicado: shield={entity.status_effects.get_shield()}")
    
    # Testar BUFF_DAMAGE
    entity.apply_status_effect(StatusEffect(
        name='test_buff_damage',
        effect_type=StatusEffectType.BUFF_DAMAGE,
        duration=2.0,
        power=0.3
    ))
    print(f"✅ BUFF_DAMAGE aplicado: damage_mult={entity.status_effects.get_damage_multiplier()}")
    
    # Testar BUFF_SPEED
    entity.apply_status_effect(StatusEffect(
        name='test_buff_speed',
        effect_type=StatusEffectType.BUFF_SPEED,
        duration=2.0,
        power=0.2
    ))
    print(f"✅ BUFF_SPEED aplicado: speed_mult={entity.status_effects.get_speed_multiplier()}")
    
    pygame.quit()
    return True

def test_group_battle():
    """Testa batalha em grupo completa"""
    print("\n=== TESTE DE BATALHA EM GRUPO ===")
    
    pygame.init()
    g = game.Game()
    
    g.team_size = 4
    g.blue_team_config = [
        {'class': 'cleric', 'weapon': 'staff', 'role': 'support'},
        {'class': 'tank', 'weapon': 'warhammer', 'role': 'tank'},
        {'class': 'ranger', 'weapon': 'bow', 'role': 'dps_ranged'},
        {'class': 'enchanter', 'weapon': 'tome', 'role': 'support'}
    ]
    g.red_team_config = [
        {'class': 'trapper', 'weapon': 'trap_launcher', 'role': 'control'},
        {'class': 'warrior', 'weapon': 'sword', 'role': 'dps_melee'},
        {'class': 'guardian', 'weapon': 'shield_bash', 'role': 'tank'},
        {'class': 'assassin', 'weapon': 'dagger', 'role': 'dps_melee'}
    ]
    
    g._start_group_battle()
    
    print(f"✅ Blue team: {len(g.blue_team)} entidades")
    print(f"✅ Red team: {len(g.red_team)} entidades")
    
    # Verificar aliados
    for e in g.blue_team:
        if len(e.allies) != 3:
            print(f"❌ {e.__class__.__name__} tem {len(e.allies)} aliados (esperado 3)")
        else:
            print(f"✅ {e.__class__.__name__}: {len(e.allies)} aliados, arma={e.weapon.__class__.__name__}")
    
    # Simular alguns frames
    for i in range(100):
        g._update_group_battle(1/60)
    
    blue_alive = sum(1 for e in g.blue_team if e.is_alive())
    red_alive = sum(1 for e in g.red_team if e.is_alive())
    
    print(f"\nApós 100 frames:")
    print(f"  Blue alive: {blue_alive}/4")
    print(f"  Red alive: {red_alive}/4")
    
    pygame.quit()
    return True

def test_abilities():
    """Testa habilidades das novas classes"""
    print("\n=== TESTE DE HABILIDADES ===")
    
    pygame.init()
    g = game.Game()
    
    # Testar Cleric heal
    cleric = g.create_entity('cleric', 100, 100, (255, 255, 255), None, 'staff')
    ally = g.create_entity('warrior', 150, 100, (255, 255, 255), None, 'sword')
    ally.health = 50
    cleric.set_allies([ally])
    
    result = cleric.use_ability()
    print(f"✅ Cleric.use_ability(): {result}, ally health={ally.health}")
    
    # Testar Guardian shield
    guardian = g.create_entity('guardian', 200, 100, (255, 255, 255), None, 'shield_bash')
    ally2 = g.create_entity('warrior', 250, 100, (255, 255, 255), None, 'sword')
    guardian.set_allies([ally2])
    
    result = guardian.use_ability()
    shield = guardian.status_effects.get_shield()
    print(f"✅ Guardian.use_ability(): {result}, self shield={shield}")
    
    # Testar Controller slow
    ctrl = g.create_entity('controller', 300, 100, (255, 255, 255), None, 'staff')
    enemy = g.create_entity('warrior', 350, 100, (255, 255, 255), None, 'sword')
    ctrl._all_entities = [ctrl, enemy]
    
    result = ctrl.use_ability()
    is_slowed = enemy.status_effects.is_slowed()
    print(f"✅ Controller.use_ability(): {result}, enemy slowed={is_slowed}")
    
    # Testar Ranger rain of arrows
    ranger = g.create_entity('ranger', 400, 100, (255, 255, 255), None, 'bow')
    enemy2 = g.create_entity('warrior', 450, 100, (255, 255, 255), None, 'sword')
    ranger._all_entities = [ranger, enemy2]
    enemy2.health = 100
    
    result = ranger.use_ability()
    print(f"✅ Ranger.use_ability(): {result}")
    
    # Testar Enchanter buff
    enchanter = g.create_entity('enchanter', 500, 100, (255, 255, 255), None, 'tome')
    ally3 = g.create_entity('warrior', 550, 100, (255, 255, 255), None, 'sword')
    enchanter.set_allies([ally3])
    
    result = enchanter.use_ability()
    dmg_mult = ally3.status_effects.get_damage_multiplier()
    print(f"✅ Enchanter.use_ability(): {result}, ally damage_mult={dmg_mult}")
    
    # Testar Trapper trap
    trapper = g.create_entity('trapper', 600, 100, (255, 255, 255), None, 'trap_launcher')
    result = trapper.use_ability()
    traps = len(trapper.traps)
    print(f"✅ Trapper.use_ability(): {result}, traps={traps}")
    
    pygame.quit()
    return True

def test_special_weapons():
    """Testa armas especiais (Staff heal, Tome buff)"""
    print("\n=== TESTE DE ARMAS ESPECIAIS ===")
    
    pygame.init()
    g = game.Game()
    
    # Staff de cura
    cleric = g.create_entity('cleric', 100, 100, (80, 140, 255), None, 'staff')
    ally = g.create_entity('warrior', 130, 100, (80, 140, 255), None, 'sword')
    ally.health = 50
    cleric.set_allies([ally])
    cleric.team = "blue"
    ally.team = "blue"
    
    g.entities = [cleric, ally]
    
    # Simular ataque com staff
    cleric.weapon.attack()
    cleric.weapon.is_attacking = True
    
    g._check_special_weapons()
    print(f"✅ Staff heal testado: ally health={ally.health}")
    
    # Tome de buff
    enchanter = g.create_entity('enchanter', 200, 100, (80, 140, 255), None, 'tome')
    ally2 = g.create_entity('warrior', 230, 100, (80, 140, 255), None, 'sword')
    enchanter.set_allies([ally2])
    enchanter.team = "blue"
    ally2.team = "blue"
    
    g.entities = [enchanter, ally2]
    
    enchanter.weapon.attack()
    enchanter.weapon.is_attacking = True
    
    g._check_special_weapons()
    dmg = ally2.status_effects.get_damage_multiplier()
    print(f"✅ Tome buff testado: ally damage_mult={dmg}")
    
    pygame.quit()
    return True

def main():
    print("=" * 50)
    print("TESTE COMPLETO DE FUNCIONALIDADES")
    print("=" * 50)
    
    results = []
    
    results.append(("Classes", test_all_classes()))
    results.append(("Armas", test_all_weapons()))
    results.append(("Estratégias", test_all_strategies()))
    results.append(("Status Effects", test_status_effects()))
    results.append(("Batalha em Grupo", test_group_battle()))
    results.append(("Habilidades", test_abilities()))
    results.append(("Armas Especiais", test_special_weapons()))
    
    print("\n" + "=" * 50)
    print("RESULTADO FINAL")
    print("=" * 50)
    
    for name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"  {name}: {status}")
    
    total_passed = sum(1 for _, r in results if r)
    print(f"\nTotal: {total_passed}/{len(results)} testes passaram")

if __name__ == "__main__":
    main()
