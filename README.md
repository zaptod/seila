# ⚔️ Arena Combat - Jogo de Combate 2D com IA

Um jogo de combate 2D com física realista, múltiplas classes, armas variadas e treinamento de IA com redes neurais.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.0-red)

---

## 📋 Índice

- [Recursos](#-recursos)
- [Instalação](#-instalação)
- [Como Jogar](#-como-jogar)
- [Classes](#-classes)
- [Armas](#-armas)
- [Modos de Jogo](#-modos-de-jogo)
- [Sistema de Status](#-sistema-de-status)
- [Atualizações Recentes](#-atualizações-recentes)
- [Bugs Conhecidos](#-bugs-conhecidos)
- [Planos Futuros](#-planos-futuros)

---

## 🎮 Recursos

- **11 Classes Jogáveis** com habilidades únicas
- **10 Armas Diferentes** com mecânicas variadas
- **110 Estratégias de IA** (todas combinações classe+arma)
- **Batalhas em Grupo 5v5** com sistema de times
- **Treinamento com Rede Neural** (PPO)
- **Sistema de Status Effects** completo
- **Física realista** com colisões e knockback

---

## 🛠️ Instalação

```bash
# Clonar/baixar o projeto
cd seila

# Instalar dependências
pip install pygame torch numpy

# Executar o jogo
python game.py
```

### Requisitos
- Python 3.10+
- Pygame 2.x
- PyTorch (opcional, para treinamento NN)
- NumPy

---

## 🎯 Como Jogar

### Controles (Player 1)
| Tecla | Ação |
|-------|------|
| `W/A/S/D` | Movimento |
| `Espaço` | Atacar |
| `E` | Usar Habilidade |
| `Shift` | Dash |

### Controles (Player 2 - PVP)
| Tecla | Ação |
|-------|------|
| `Setas` | Movimento |
| `Enter` | Atacar |
| `Shift Direito` | Usar Habilidade |

### Controles Gerais
| Tecla | Ação |
|-------|------|
| `ESC` | Menu/Voltar |
| `P` | Pausar |
| `R` | Reiniciar |
| `1-9` | Velocidade de simulação |

---

## ⚔️ Classes

### Classes Originais (5)

| Classe | Tipo | Habilidade | Descrição |
|--------|------|------------|-----------|
| **Warrior** | DPS Melee | Fúria | Aumenta dano e velocidade por 5s |
| **Berserker** | DPS Melee | Rage | Quanto menos vida, mais dano |
| **Assassin** | DPS Melee | Invisibilidade | Fica invisível e próximo ataque dá crítico |
| **Tank** | Tank | Fortaleza | Reduz dano recebido em 50% por 4s |
| **Lancer** | DPS Melee | Investida | Dash longo que causa dano |

### Novas Classes (6) ✨

| Classe | Tipo | Habilidade | Descrição |
|--------|------|------------|-----------|
| **Cleric** | Suporte | Cura Divina | Cura aliados próximos + HoT |
| **Guardian** | Tank | Escudo Protetor | Dá escudo a todos aliados |
| **Controller** | Controle | Campo de Lentidão | Slow em área nos inimigos |
| **Ranger** | DPS Ranged | Chuva de Flechas | Dano em área à distância |
| **Enchanter** | Suporte | Bênção de Guerra | Buff de dano e velocidade |
| **Trapper** | Controle | Armadilha | Coloca armadilha que dá root |

---

## 🗡️ Armas

### Armas Originais (4)

| Arma | Tipo | Dano | Velocidade | Especial |
|------|------|------|------------|----------|
| **Sword** | Melee | Médio | Média | Balanceada |
| **Greatsword** | Melee | Alto | Lenta | Alto knockback |
| **Dagger** | Melee | Baixo | Rápida | Alto crítico |
| **Spear** | Melee | Médio | Média | Longo alcance |

### Novas Armas (6) ✨

| Arma | Tipo | Dano | Velocidade | Especial |
|------|------|------|------------|----------|
| **Staff** | Suporte | Baixo | Média | **Cura aliados** ao atacar |
| **Bow** | Ranged | Médio | Média | **Projéteis** de longo alcance |
| **Warhammer** | Melee | Alto | Lenta | **Stun** ao acertar |
| **Tome** | Suporte | Baixo | Média | **Buff** aliados ao atacar |
| **Shield Bash** | Melee | Baixo | Média | **Slow** + alto knockback |
| **Trap Launcher** | Ranged | Médio | Lenta | Lança **armadilhas** que dão root |

---

## 🎲 Modos de Jogo

### 1. PVE (Player vs AI)
Enfrente uma IA controlada com estratégias específicas para cada combinação classe+arma.

### 2. PVP (Player vs Player)
Batalha local entre dois jogadores.

### 3. AI vs AI
Assista duas IAs lutando. Útil para testar estratégias e balanceamento.

### 4. Batalha em Grupo (5v5) ✨
Times de 5 entidades cada. Suporta:
- Configuração de classes e armas por membro
- Sistema de aliados para habilidades de suporte
- Roles: Tank, DPS Melee, DPS Ranged, Suporte, Controle

### 5. Treinamento com Rede Neural
Treine uma IA usando PPO (Proximal Policy Optimization):
- Observações: posição, vida, cooldowns, distância ao inimigo
- Ações: movimento, ataque, habilidade, dash
- Rewards customizados por dano causado/recebido, vitória, etc.

---

## 💫 Sistema de Status

| Status | Efeito | Fontes |
|--------|--------|--------|
| **STUN** | Não pode agir | Warhammer |
| **SLOW** | Velocidade reduzida | Shield Bash, Controller |
| **ROOT** | Não pode mover | Trapper, Trap Launcher |
| **SILENCE** | Não pode usar habilidade | - |
| **KNOCKUP** | Levantado no ar | - |
| **SHIELD** | Absorve dano | Guardian |
| **BUFF_DAMAGE** | +X% dano | Enchanter, Tome |
| **BUFF_SPEED** | +X% velocidade | Enchanter |
| **HEAL_OVER_TIME** | Cura por segundo | Cleric |
| **MARKED** | +50% dano recebido | - |

---

## 🔄 Atualizações Recentes

### Versão 2.0 (Dezembro 2024)

#### Novas Classes
- ✅ **Cleric** - Curandeiro com cura em área e HoT
- ✅ **Guardian** - Tank que dá escudo aos aliados
- ✅ **Controller** - Mago de controle com slow em área
- ✅ **Ranger** - Arqueiro com chuva de flechas
- ✅ **Enchanter** - Buffer que aumenta dano e velocidade
- ✅ **Trapper** - Especialista em armadilhas

#### Novas Armas
- ✅ **Staff (Cajado)** - Cura aliados próximos ao atacar
- ✅ **Bow (Arco)** - Dispara flechas como projéteis
- ✅ **Warhammer (Martelo)** - Causa stun ao acertar
- ✅ **Tome (Tomo)** - Buffa aliados ao atacar
- ✅ **Shield Bash (Escudo)** - Causa slow e alto knockback
- ✅ **Trap Launcher** - Lança armadilhas que dão root

#### Sistema de Batalha em Grupo
- ✅ Batalhas 5v5 integradas
- ✅ Sistema de times (azul/vermelho)
- ✅ Configuração de aliados para habilidades de suporte
- ✅ Roles específicos (tank, dps, support, control)

#### IA e Estratégias
- ✅ 110 estratégias de IA (todas 11 classes × 10 armas)
- ✅ Estratégias específicas para cada combinação
- ✅ IA de suporte que cura/buffa aliados
- ✅ IA de controle que prioriza CC

#### Correções de Bugs
- ✅ Trapper root não aplicava - corrigido
- ✅ TrapLauncher era melee - reescrito como projétil
- ✅ Hitbox types (line, circle, projectile) implementados
- ✅ Stun/Slow de armas não aplicava - corrigido
- ✅ Controller/Ranger abilities precisavam de lista de inimigos - corrigido
- ✅ Staff heal e Tome buff não funcionavam - implementado `_check_special_weapons()`
- ✅ Aliados não configurados em batalha de grupo - corrigido

#### Novos Métodos em StatusEffectManager
- ✅ `is_slowed()` - Verifica se está com slow
- ✅ `get_shield()` - Retorna quantidade de escudo
- ✅ `get_speed_multiplier()` - Multiplicador de velocidade com buffs/debuffs

---

## 🐛 Bugs Conhecidos

### Prioridade Média
1. **Slow motion no treinamento** - Às vezes trava em slow motion
2. **Múltiplos status do mesmo tipo** - Podem se sobrepor de forma inconsistente

### Prioridade Baixa
3. **Renderização de armadilhas** - Armadilhas do Trapper e TrapLauncher podem sobrepor
4. **Som** - Não há efeitos sonoros implementados
5. **Animações** - Animações são simples, poderiam ser melhoradas

---

## 🚀 Planos Futuros

### Curto Prazo (v2.1) ✅ CONCLUÍDO
- [x] **Balanceamento** - Stats ajustados para todas as 11 classes e 10 armas
- [x] **Colisão de projéteis** - Flechas e armadilhas colidem com bordas da arena
- [x] **UI melhorada** - Barras de vida com HP, escudo, cooldown e ícones de status
- [x] **Indicadores visuais** - Preview de área de habilidades quando prontas

### Médio Prazo (v2.5)
- [ ] **Mais mapas** - Arenas com obstáculos e layouts diferentes
- [ ] **Sistema de itens** - Equipamentos que modificam stats
- [ ] **Modo história** - Campanha single-player com progressão
- [ ] **Efeitos sonoros** - Sons para ataques, habilidades, hits
- [ ] **Música** - Trilha sonora para menus e batalhas

### Longo Prazo (v3.0)
- [ ] **Multiplayer online** - Batalhas PVP pela internet
- [ ] **Mais classes** - Necromancer, Paladin, Monk, etc.
- [ ] **Sistema de skills** - Árvore de habilidades por classe
- [ ] **Ranking/ELO** - Sistema competitivo
- [ ] **Editor de mapas** - Criar arenas customizadas
- [ ] **Replays** - Salvar e assistir partidas

### Melhorias de IA
- [ ] **Meta-learning** - IA que aprende a jogar contra diferentes oponentes
- [ ] **Curriculum learning** - Treinamento progressivo contra IAs mais difíceis
- [ ] **Multi-agent training** - Treinar times inteiros de IA
- [ ] **Imitation learning** - IA que aprende observando jogadores humanos

---

## 📁 Estrutura do Projeto

```
seila/
├── game.py              # Arquivo principal, loop de jogo
├── entities.py          # Classes de entidades (Warrior, Cleric, etc.)
├── weapons.py           # Classes de armas (Sword, Bow, etc.)
├── controller.py        # Controladores de IA e estratégias
├── physics.py           # Sistema de física e colisões
├── stats.py             # Status effects e gerenciamento de stats
├── game_state.py        # Estado do jogo
├── maps.py              # Configuração de arenas
├── balance_config.py    # Configurações centralizadas de balanceamento
├── config_db.py         # Configurações do banco de dados
├── train.py             # Treinamento de IA
├── team_train.py        # Treinamento de times
├── tournament_ui.py     # Interface de torneios
├── test_all_features.py # Testes de funcionalidades
├── test_all_modes.py    # Testes de modos de jogo
├── custom_config.json   # Configurações customizadas
├── models/              # Modelos de IA treinados
└── README.md            # Este arquivo
```

---

## 🧪 Executando Testes

```bash
# Testar todas as funcionalidades
python test_all_features.py

# Testar todos os modos de jogo
python test_all_modes.py
```

---

## 📜 Licença

Este projeto é para fins educacionais e de entretenimento.

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:
- Reportar bugs
- Sugerir novas features
- Melhorar o balanceamento
- Adicionar novas classes/armas

---

**Desenvolvido com ❤️ e Python**
