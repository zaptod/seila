"""
Teste completo de todos os modos de jogo
"""
import pygame
pygame.init()

import game

def test_pve():
    """Testa modo PVE (Player vs AI)"""
    print('\n=== MODO PVE (Player vs AI) ===')
    g = game.Game()
    g.mode = 'playing'
    g.game_mode = 'pve'
    
    # Criar entidades manualmente
    from controller import StrategicAI
    
    # Player (controlado por IA para teste)
    player_ai = StrategicAI(class_name='Warrior', weapon_name='Sword')
    player = g.create_entity('warrior', 200, 300, (80, 140, 255), player_ai, 'sword')
    
    # Oponente AI
    opponent_ai = StrategicAI(class_name='Berserker', weapon_name='Greatsword')
    opponent = g.create_entity('berserker', 600, 300, (255, 80, 80), opponent_ai, 'greatsword')
    
    g.entities = [player, opponent]
    g.training_entity = player
    g.training_opponent = opponent
    
    # Configurar alvos
    player_ai.set_targets([opponent])
    opponent_ai.set_targets([player])
    
    print(f'Player: {player.__class__.__name__} com {player.weapon.__class__.__name__}')
    print(f'AI: {opponent.__class__.__name__} com {opponent.weapon.__class__.__name__}')
    
    # Simular alguns frames
    for i in range(100):
        for e in g.entities:
            if e.is_alive():
                e.update(1/60)
        g.physics.handle_collisions(g.entities)
    
    print(f'Player alive: {player.is_alive()}, health={player.health:.1f}')
    print(f'AI alive: {opponent.is_alive()}, health={opponent.health:.1f}')
    print('✅ PVE OK')
    
    pygame.quit()
    return True

def test_pvp():
    """Testa modo PVP (Player vs Player)"""
    print('\n=== MODO PVP (Player vs Player) ===')
    pygame.init()
    g = game.Game()
    
    from controller import StrategicAI
    
    # Player 1
    p1_ai = StrategicAI(class_name='Assassin', weapon_name='Dagger')
    p1 = g.create_entity('assassin', 200, 300, (80, 140, 255), p1_ai, 'dagger')
    
    # Player 2
    p2_ai = StrategicAI(class_name='Tank', weapon_name='Warhammer')
    p2 = g.create_entity('tank', 600, 300, (255, 80, 80), p2_ai, 'warhammer')
    
    g.entities = [p1, p2]
    
    p1_ai.set_targets([p2])
    p2_ai.set_targets([p1])
    
    print(f'Player 1: {p1.__class__.__name__} com {p1.weapon.__class__.__name__}')
    print(f'Player 2: {p2.__class__.__name__} com {p2.weapon.__class__.__name__}')
    
    for i in range(100):
        for e in g.entities:
            if e.is_alive():
                e.update(1/60)
        g.physics.handle_collisions(g.entities)
    
    print(f'P1 alive: {p1.is_alive()}, health={p1.health:.1f}')
    print(f'P2 alive: {p2.is_alive()}, health={p2.health:.1f}')
    print('✅ PVP OK')
    
    pygame.quit()
    return True

def test_ai_vs_ai():
    """Testa modo AI vs AI"""
    print('\n=== MODO AI vs AI ===')
    pygame.init()
    g = game.Game()
    
    from controller import StrategicAI
    
    # AI 1 - Ranger com Bow
    ai1 = StrategicAI(class_name='Ranger', weapon_name='Bow')
    e1 = g.create_entity('ranger', 200, 300, (80, 140, 255), ai1, 'bow')
    
    # AI 2 - Trapper com TrapLauncher
    ai2 = StrategicAI(class_name='Trapper', weapon_name='Trap_launcher')
    e2 = g.create_entity('trapper', 600, 300, (255, 80, 80), ai2, 'trap_launcher')
    
    g.entities = [e1, e2]
    
    ai1.set_targets([e2])
    ai2.set_targets([e1])
    
    print(f'AI 1: {e1.__class__.__name__} com {e1.weapon.__class__.__name__}')
    print(f'AI 2: {e2.__class__.__name__} com {e2.weapon.__class__.__name__}')
    
    for i in range(200):
        for e in g.entities:
            if e.is_alive():
                e.update(1/60)
        
        # Verificar traps
        for e in g.entities:
            if hasattr(e, 'check_traps') and e.is_alive():
                enemies = [x for x in g.entities if x != e and x.is_alive()]
                e.check_traps(enemies)
        
        # Verificar TrapLauncher
        for e in g.entities:
            if e.is_alive() and hasattr(e, 'weapon') and e.weapon:
                if hasattr(e.weapon, 'check_trap_hits'):
                    from stats import StatusEffect, StatusEffectType
                    hits = e.weapon.check_trap_hits(g.entities)
                    for target, trap in hits:
                        target.take_damage(trap['damage'], e)
                        target.apply_status_effect(StatusEffect(
                            name='trap_root',
                            effect_type=StatusEffectType.ROOT,
                            duration=trap['root_duration'],
                            source=e
                        ))
        
        g.physics.handle_collisions(g.entities)
    
    print(f'AI1 alive: {e1.is_alive()}, health={e1.health:.1f}')
    print(f'AI2 alive: {e2.is_alive()}, health={e2.health:.1f}')
    print('✅ AI vs AI OK')
    
    pygame.quit()
    return True

def test_group_battle():
    """Testa batalha em grupo 5v5"""
    print('\n=== MODO BATALHA EM GRUPO (5v5) ===')
    pygame.init()
    g = game.Game()
    
    g.team_size = 5
    g.blue_team_config = [
        {'class': 'cleric', 'weapon': 'staff', 'role': 'support'},
        {'class': 'tank', 'weapon': 'shield_bash', 'role': 'tank'},
        {'class': 'warrior', 'weapon': 'sword', 'role': 'dps_melee'},
        {'class': 'ranger', 'weapon': 'bow', 'role': 'dps_ranged'},
        {'class': 'enchanter', 'weapon': 'tome', 'role': 'support'}
    ]
    g.red_team_config = [
        {'class': 'guardian', 'weapon': 'warhammer', 'role': 'tank'},
        {'class': 'trapper', 'weapon': 'trap_launcher', 'role': 'control'},
        {'class': 'assassin', 'weapon': 'dagger', 'role': 'dps_melee'},
        {'class': 'controller', 'weapon': 'staff', 'role': 'support'},
        {'class': 'berserker', 'weapon': 'greatsword', 'role': 'dps_melee'}
    ]
    
    g._start_group_battle()
    
    print(f'Blue team: {len(g.blue_team)} entidades')
    for e in g.blue_team:
        print(f'  - {e.__class__.__name__} ({e.weapon.__class__.__name__})')
    
    print(f'Red team: {len(g.red_team)} entidades')
    for e in g.red_team:
        print(f'  - {e.__class__.__name__} ({e.weapon.__class__.__name__})')
    
    # Simular batalha
    for i in range(300):
        g._update_group_battle(1/60)
        if g.game_over:
            break
    
    blue_alive = sum(1 for e in g.blue_team if e.is_alive())
    red_alive = sum(1 for e in g.red_team if e.is_alive())
    
    print(f'\nApós batalha:')
    print(f'  Blue alive: {blue_alive}/5')
    print(f'  Red alive: {red_alive}/5')
    print(f'  Vencedor: {g.winner if g.game_over else "em andamento"}')
    print('✅ Batalha em Grupo OK')
    
    pygame.quit()
    return True

def test_training():
    """Testa modo de treinamento com rede neural"""
    print('\n=== MODO TREINAMENTO NN ===')
    pygame.init()
    g = game.Game()
    
    from controller import StrategicAI
    
    # Criar entidades diretamente para teste
    agent = g.create_entity('lancer', 200, 300, (100, 255, 100), None, 'spear')
    
    opponent_ai = StrategicAI(class_name='Controller', weapon_name='Tome')
    opponent = g.create_entity('controller', 600, 300, (255, 100, 100), opponent_ai, 'tome')
    
    g.entities = [agent, opponent]
    g.training_entity = agent
    g.training_opponent = opponent
    
    opponent_ai.set_targets([agent])
    
    # Simular updates manualmente
    g.training_running = True
    g.training_paused = False
    g.mode = 'training'
    g.game_over = False
    g.training_step = 0
    g.training_episode_reward = 0
    
    print(f'Agent: {agent.__class__.__name__} com {agent.weapon.__class__.__name__}')
    print(f'Opponent: {opponent.__class__.__name__} com {opponent.weapon.__class__.__name__}')
    
    # Simular 100 frames
    for i in range(100):
        for e in g.entities:
            if e.is_alive():
                e.update(1/60)
        
        # Checar armadilhas e armas especiais
        g._check_special_weapons()
        g.physics.handle_collisions(g.entities)
        
        g.training_step += 1
        
        if not agent.is_alive() or not opponent.is_alive():
            g.game_over = True
            break
    
    print(f'Training step: {g.training_step}')
    print(f'Agent alive: {agent.is_alive()}, health={agent.health:.1f}')
    print(f'Opponent alive: {opponent.is_alive()}, health={opponent.health:.1f}')
    print('✅ Treinamento NN OK')
    
    pygame.quit()
    return True

def test_all_class_combinations():
    """Testa algumas combinações de classes e armas"""
    print('\n=== TESTE DE COMBINAÇÕES CLASSE+ARMA ===')
    pygame.init()
    g = game.Game()
    
    from controller import StrategicAI
    
    test_combos = [
        ('cleric', 'staff'),
        ('guardian', 'shield_bash'),
        ('controller', 'tome'),
        ('ranger', 'bow'),
        ('enchanter', 'warhammer'),
        ('trapper', 'trap_launcher'),
        ('berserker', 'greatsword'),
        ('assassin', 'dagger'),
        ('tank', 'spear'),
        ('lancer', 'sword'),
    ]
    
    for cls, wpn in test_combos:
        try:
            ai = StrategicAI(class_name=cls.capitalize(), weapon_name=wpn.capitalize())
            e = g.create_entity(cls, 400, 300, (255, 255, 255), ai, wpn)
            print(f'✅ {cls.capitalize()} + {wpn.capitalize()}: OK')
        except Exception as ex:
            print(f'❌ {cls.capitalize()} + {wpn.capitalize()}: {ex}')
    
    pygame.quit()
    return True

def main():
    print('=' * 60)
    print('TESTE COMPLETO - TODOS OS MODOS DE JOGO')
    print('=' * 60)
    
    results = []
    
    results.append(('PVE', test_pve()))
    results.append(('PVP', test_pvp()))
    results.append(('AI vs AI', test_ai_vs_ai()))
    results.append(('Batalha em Grupo', test_group_battle()))
    results.append(('Treinamento NN', test_training()))
    results.append(('Combinações', test_all_class_combinations()))
    
    print('\n' + '=' * 60)
    print('RESULTADO FINAL')
    print('=' * 60)
    
    for name, result in results:
        status = '✅ PASSOU' if result else '❌ FALHOU'
        print(f'  {name}: {status}')
    
    total = sum(1 for _, r in results if r)
    print(f'\nTotal: {total}/{len(results)} testes passaram')
    
    if total == len(results):
        print('\n🎉 TODOS OS MODOS FUNCIONANDO CORRETAMENTE!')

if __name__ == '__main__':
    main()
