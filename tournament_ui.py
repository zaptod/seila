"""
Interface Visual do Sistema de Torneio
======================================
Menu visual para configurar equipes, batalhas em grupo e torneios.
"""

import pygame
import math
import sys
import os
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Imports do jogo
from entities import ClassRegistry, Entity
from weapons import WeaponRegistry
from physics import Physics
from controller import StrategicAI
from game_state import GameState, ArenaConfig, GameConfig
from team_train import (
    TeamBattleEnv, TeamRole, TrainingMode, 
    TEAM_COMPOSITIONS, TeamMember, RewardCalculator
)


# ============================================================================
# CORES E ESTILOS
# ============================================================================

class Colors:
    """Paleta de cores do jogo"""
    # Fundos
    BG_DARK = (20, 20, 30)
    BG_PANEL = (35, 35, 50)
    BG_HOVER = (50, 50, 70)
    BG_SELECTED = (60, 80, 120)
    
    # Times
    BLUE_TEAM = (80, 140, 255)
    BLUE_TEAM_DARK = (50, 90, 180)
    RED_TEAM = (255, 80, 80)
    RED_TEAM_DARK = (180, 50, 50)
    
    # Papéis
    ROLE_HEALER = (100, 255, 150)
    ROLE_TANK = (150, 150, 200)
    ROLE_DPS = (255, 150, 100)
    ROLE_CONTROLLER = (200, 100, 255)
    ROLE_SUPPORT = (255, 220, 100)
    
    # UI
    WHITE = (255, 255, 255)
    GRAY = (150, 150, 150)
    DARK_GRAY = (80, 80, 80)
    GREEN = (100, 255, 100)
    YELLOW = (255, 255, 100)
    
    @staticmethod
    def get_role_color(role: TeamRole) -> Tuple[int, int, int]:
        """Retorna cor baseada no papel"""
        colors = {
            TeamRole.HEALER: Colors.ROLE_HEALER,
            TeamRole.TANK: Colors.ROLE_TANK,
            TeamRole.DPS_MELEE: Colors.ROLE_DPS,
            TeamRole.DPS_RANGED: Colors.ROLE_DPS,
            TeamRole.CONTROLLER: Colors.ROLE_CONTROLLER,
            TeamRole.SUPPORT: Colors.ROLE_SUPPORT,
        }
        return colors.get(role, Colors.WHITE)


# ============================================================================
# COMPONENTES DE UI
# ============================================================================

class Button:
    """Botão clicável"""
    
    def __init__(self, x: int, y: int, width: int, height: int, 
                 text: str, color: Tuple[int, int, int] = Colors.BG_PANEL):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.hover_color = Colors.BG_HOVER
        self.selected_color = Colors.BG_SELECTED
        self.is_hovered = False
        self.is_selected = False
        self.enabled = True
    
    def update(self, mouse_pos: Tuple[int, int]):
        """Atualiza estado de hover"""
        self.is_hovered = self.rect.collidepoint(mouse_pos) and self.enabled
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font):
        """Desenha o botão"""
        if not self.enabled:
            color = Colors.DARK_GRAY
        elif self.is_selected:
            color = self.selected_color
        elif self.is_hovered:
            color = self.hover_color
        else:
            color = self.color
        
        # Fundo
        pygame.draw.rect(screen, color, self.rect, border_radius=8)
        pygame.draw.rect(screen, Colors.GRAY, self.rect, 2, border_radius=8)
        
        # Texto
        text_color = Colors.WHITE if self.enabled else Colors.DARK_GRAY
        text_surf = font.render(self.text, True, text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)
    
    def is_clicked(self, mouse_pos: Tuple[int, int], mouse_pressed: bool) -> bool:
        """Verifica se foi clicado"""
        return self.enabled and mouse_pressed and self.rect.collidepoint(mouse_pos)


class ClassCard:
    """Card de seleção de classe"""
    
    def __init__(self, x: int, y: int, class_name: str, weapon_name: str, role: TeamRole):
        self.rect = pygame.Rect(x, y, 140, 180)
        self.class_name = class_name
        self.weapon_name = weapon_name
        self.role = role
        self.is_selected = False
        self.is_hovered = False
    
    def update(self, mouse_pos: Tuple[int, int]):
        self.is_hovered = self.rect.collidepoint(mouse_pos)
    
    def draw(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font):
        # Fundo
        if self.is_selected:
            color = Colors.BG_SELECTED
        elif self.is_hovered:
            color = Colors.BG_HOVER
        else:
            color = Colors.BG_PANEL
        
        pygame.draw.rect(screen, color, self.rect, border_radius=10)
        
        # Borda colorida pelo papel
        role_color = Colors.get_role_color(self.role)
        pygame.draw.rect(screen, role_color, self.rect, 3, border_radius=10)
        
        # Ícone de círculo representando a classe
        icon_center = (self.rect.centerx, self.rect.y + 50)
        pygame.draw.circle(screen, role_color, icon_center, 30)
        pygame.draw.circle(screen, Colors.WHITE, icon_center, 30, 2)
        
        # Nome da classe
        text = font.render(self.class_name.title(), True, Colors.WHITE)
        screen.blit(text, (self.rect.centerx - text.get_width()//2, self.rect.y + 90))
        
        # Arma
        weapon_text = small_font.render(self.weapon_name.title(), True, Colors.GRAY)
        screen.blit(weapon_text, (self.rect.centerx - weapon_text.get_width()//2, self.rect.y + 115))
        
        # Papel
        role_text = small_font.render(self.role.value.upper(), True, role_color)
        screen.blit(role_text, (self.rect.centerx - role_text.get_width()//2, self.rect.y + 150))


# ============================================================================
# TELAS DO SISTEMA DE TORNEIO
# ============================================================================

class TournamentUI:
    """
    Interface principal do sistema de torneio.
    Gerencia todos os menus e visualizações.
    """
    
    def __init__(self, screen_width: int, screen_height: int):
        self.width = screen_width
        self.height = screen_height
        
        # Estado
        self.current_screen = "main_menu"  # main_menu, team_setup, battle, tournament, results
        self.running = True
        
        # Pygame
        self.screen = None
        self.clock = None
        self.fonts = {}
        
        # Configuração de batalha
        self.team_size = 2  # 2, 3, ou 5
        self.blue_team_config = []
        self.red_team_config = []
        
        # Seleção atual
        self.selecting_team = "blue"  # "blue" ou "red"
        self.selected_slot = 0
        
        # Ambiente de batalha
        self.battle_env: Optional[TeamBattleEnv] = None
        self.battle_running = False
        self.battle_speed = 1  # 1x, 2x, 4x
        
        # Torneio
        self.tournament_teams = []
        self.tournament_results = []
        self.tournament_round = 0
        
        # UI Elements
        self.buttons = {}
        self.class_cards = []
        
        # Classes e armas disponíveis
        self.available_classes = ClassRegistry.list_classes()
        self.available_weapons = WeaponRegistry.list_weapons()
        
        # Combinações válidas por papel
        self.role_combos = {
            TeamRole.HEALER: [
                ("cleric", "staff"),
                ("enchanter", "tome"),
            ],
            TeamRole.TANK: [
                ("guardian", "shield_bash"),
                ("tank", "spear"),
                ("tank", "sword"),
            ],
            TeamRole.DPS_MELEE: [
                ("warrior", "sword"),
                ("berserker", "greatsword"),
                ("assassin", "dagger"),
                ("lancer", "spear"),
            ],
            TeamRole.DPS_RANGED: [
                ("ranger", "bow"),
            ],
            TeamRole.CONTROLLER: [
                ("controller", "warhammer"),
                ("trapper", "trap_launcher"),
            ],
            TeamRole.SUPPORT: [
                ("enchanter", "tome"),
                ("cleric", "staff"),
            ],
        }
        
        # Estatísticas da sessão
        self.session_stats = {
            "battles_played": 0,
            "blue_wins": 0,
            "red_wins": 0,
            "draws": 0
        }
    
    def initialize(self):
        """Inicializa pygame e recursos"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Circle Warriors - Torneio em Grupo")
        self.clock = pygame.time.Clock()
        
        # Fontes
        self.fonts = {
            'small': pygame.font.Font(None, 20),
            'medium': pygame.font.Font(None, 28),
            'large': pygame.font.Font(None, 40),
            'title': pygame.font.Font(None, 56),
        }
        
        # Criar botões do menu principal
        self._create_main_menu_buttons()
    
    def _create_main_menu_buttons(self):
        """Cria botões do menu principal"""
        center_x = self.width // 2
        start_y = 250
        btn_width = 300
        btn_height = 50
        spacing = 70
        
        self.buttons['main_menu'] = [
            Button(center_x - btn_width//2, start_y, btn_width, btn_height, 
                   "⚔️ Batalha 2v2", Colors.BLUE_TEAM_DARK),
            Button(center_x - btn_width//2, start_y + spacing, btn_width, btn_height,
                   "⚔️ Batalha 3v3", Colors.BLUE_TEAM_DARK),
            Button(center_x - btn_width//2, start_y + spacing*2, btn_width, btn_height,
                   "⚔️ Batalha 5v5", Colors.BLUE_TEAM_DARK),
            Button(center_x - btn_width//2, start_y + spacing*3, btn_width, btn_height,
                   "🏆 Torneio", Colors.ROLE_SUPPORT),
            Button(center_x - btn_width//2, start_y + spacing*4, btn_width, btn_height,
                   "📊 Estatísticas", Colors.BG_PANEL),
            Button(center_x - btn_width//2, start_y + spacing*5, btn_width, btn_height,
                   "❌ Sair", Colors.RED_TEAM_DARK),
        ]
    
    def run(self):
        """Loop principal"""
        self.initialize()
        
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)
        
        pygame.quit()
    
    def handle_events(self):
        """Processa eventos"""
        mouse_pos = pygame.mouse.get_pos()
        mouse_clicked = False
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.VIDEORESIZE:
                self.width = event.w
                self.height = event.h
                self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                self._create_main_menu_buttons()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_clicked = True
            
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event.key)
        
        # Atualizar UI baseado na tela atual
        if self.current_screen == "main_menu":
            self._update_main_menu(mouse_pos, mouse_clicked)
        elif self.current_screen == "team_setup":
            self._update_team_setup(mouse_pos, mouse_clicked)
        elif self.current_screen == "composition_select":
            self._update_composition_select(mouse_pos, mouse_clicked)
        elif self.current_screen == "battle":
            self._update_battle(mouse_pos, mouse_clicked)
    
    def _handle_key(self, key):
        """Processa teclas"""
        if key == pygame.K_ESCAPE:
            if self.current_screen == "main_menu":
                self.running = False
            elif self.current_screen == "battle":
                self._end_battle()
            else:
                self.current_screen = "main_menu"
        
        elif key == pygame.K_SPACE:
            if self.current_screen == "battle" and self.battle_env:
                self.battle_running = not self.battle_running
        
        elif key == pygame.K_1:
            if self.current_screen == "battle":
                self.battle_speed = 1
        elif key == pygame.K_2:
            if self.current_screen == "battle":
                self.battle_speed = 2
        elif key == pygame.K_4:
            if self.current_screen == "battle":
                self.battle_speed = 4
    
    def _update_main_menu(self, mouse_pos, mouse_clicked):
        """Atualiza menu principal"""
        for i, btn in enumerate(self.buttons['main_menu']):
            btn.update(mouse_pos)
            
            if btn.is_clicked(mouse_pos, mouse_clicked):
                if i == 0:  # 2v2
                    self.team_size = 2
                    self._start_team_setup()
                elif i == 1:  # 3v3
                    self.team_size = 3
                    self._start_team_setup()
                elif i == 2:  # 5v5
                    self.team_size = 5
                    self._start_team_setup()
                elif i == 3:  # Torneio
                    self._start_tournament_setup()
                elif i == 4:  # Estatísticas
                    self.current_screen = "stats"
                elif i == 5:  # Sair
                    self.running = False
    
    def _start_team_setup(self):
        """Inicia configuração de equipes"""
        self.current_screen = "composition_select"
        self.selecting_team = "blue"
        self.blue_team_config = []
        self.red_team_config = []
        self._create_composition_buttons()
    
    def _create_composition_buttons(self):
        """Cria botões de seleção de composição"""
        compositions = self._get_compositions_for_size(self.team_size)
        
        self.buttons['compositions'] = []
        start_y = 200
        btn_height = 80
        spacing = 90
        btn_width = 400
        center_x = self.width // 2
        
        for i, (name, config) in enumerate(compositions.items()):
            btn = Button(
                center_x - btn_width//2,
                start_y + i * spacing,
                btn_width, btn_height,
                name.replace("_", " ").title(),
                Colors.BLUE_TEAM_DARK if self.selecting_team == "blue" else Colors.RED_TEAM_DARK
            )
            self.buttons['compositions'].append((btn, config, name))
    
    def _get_compositions_for_size(self, size: int) -> Dict:
        """Retorna composições disponíveis para o tamanho"""
        prefix = f"{size}v{size}_"
        return {k: v for k, v in TEAM_COMPOSITIONS.items() if k.startswith(prefix)}
    
    def _update_composition_select(self, mouse_pos, mouse_clicked):
        """Atualiza seleção de composição"""
        if 'compositions' not in self.buttons:
            return
        
        for btn, config, name in self.buttons['compositions']:
            btn.update(mouse_pos)
            
            if btn.is_clicked(mouse_pos, mouse_clicked):
                if self.selecting_team == "blue":
                    self.blue_team_config = config.copy()
                    self.selecting_team = "red"
                    self._create_composition_buttons()
                else:
                    self.red_team_config = config.copy()
                    self._start_battle()
    
    def _start_battle(self):
        """Inicia a batalha"""
        self.current_screen = "battle"
        
        # Determinar tamanho da arena baseado no tamanho da equipe
        arena_sizes = {
            2: (1000, 700),
            3: (1200, 800),
            5: (1400, 900),
        }
        arena_size = arena_sizes.get(self.team_size, (1200, 800))
        
        # Criar ambiente
        self.battle_env = TeamBattleEnv(
            team_size=self.team_size,
            blue_team_config=self.blue_team_config,
            red_team_config=self.red_team_config,
            render_mode=None,  # Vamos renderizar manualmente
            max_steps=6000,
            arena_size=arena_size
        )
        
        self.battle_env.reset()
        self.battle_running = True
        self.battle_speed = 1
    
    def _update_battle(self, mouse_pos, mouse_clicked):
        """Atualiza a batalha"""
        if not self.battle_env or not self.battle_running:
            return
        
        # Executar steps baseado na velocidade
        for _ in range(self.battle_speed):
            # Ações aleatórias para demonstração (substituir por IA/NN)
            actions = [np.random.uniform(-1, 1, 4) for _ in range(self.team_size)]
            obs, rewards, terminated, truncated, info = self.battle_env.step(actions)
            
            if terminated or truncated:
                self._end_battle_round(info)
                break
    
    def _end_battle_round(self, info):
        """Finaliza uma rodada de batalha"""
        self.battle_running = False
        self.session_stats["battles_played"] += 1
        
        if info["blue_alive"] > info["red_alive"]:
            self.session_stats["blue_wins"] += 1
        elif info["red_alive"] > info["blue_alive"]:
            self.session_stats["red_wins"] += 1
        else:
            self.session_stats["draws"] += 1
    
    def _end_battle(self):
        """Encerra a batalha e volta ao menu"""
        self.battle_env = None
        self.battle_running = False
        self.current_screen = "main_menu"
    
    def _start_tournament_setup(self):
        """Inicia configuração do torneio"""
        self.team_size = 2  # Torneio padrão 2v2
        self.tournament_teams = []
        self.current_screen = "tournament_setup"
    
    def _update_team_setup(self, mouse_pos, mouse_clicked):
        """Atualiza tela de setup de equipe manual"""
        pass  # Implementar se quiser seleção manual de membros
    
    def update(self):
        """Atualiza lógica do jogo"""
        pass
    
    def draw(self):
        """Desenha a tela atual"""
        self.screen.fill(Colors.BG_DARK)
        
        if self.current_screen == "main_menu":
            self._draw_main_menu()
        elif self.current_screen == "composition_select":
            self._draw_composition_select()
        elif self.current_screen == "battle":
            self._draw_battle()
        elif self.current_screen == "stats":
            self._draw_stats()
        elif self.current_screen == "tournament_setup":
            self._draw_tournament_setup()
        
        pygame.display.flip()
    
    def _draw_main_menu(self):
        """Desenha menu principal"""
        # Título
        title = self.fonts['title'].render("⚔️ CIRCLE WARRIORS ⚔️", True, Colors.WHITE)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 80))
        
        subtitle = self.fonts['large'].render("Sistema de Torneio em Grupo", True, Colors.GRAY)
        self.screen.blit(subtitle, (self.width//2 - subtitle.get_width()//2, 150))
        
        # Botões
        for btn in self.buttons['main_menu']:
            btn.draw(self.screen, self.fonts['medium'])
        
        # Estatísticas da sessão
        stats_y = self.height - 100
        stats_text = f"Sessão: {self.session_stats['battles_played']} batalhas | " \
                    f"Azul: {self.session_stats['blue_wins']} | " \
                    f"Vermelho: {self.session_stats['red_wins']} | " \
                    f"Empates: {self.session_stats['draws']}"
        stats_surf = self.fonts['small'].render(stats_text, True, Colors.GRAY)
        self.screen.blit(stats_surf, (self.width//2 - stats_surf.get_width()//2, stats_y))
    
    def _draw_composition_select(self):
        """Desenha seleção de composição"""
        # Título
        team_name = "TIME AZUL" if self.selecting_team == "blue" else "TIME VERMELHO"
        team_color = Colors.BLUE_TEAM if self.selecting_team == "blue" else Colors.RED_TEAM
        
        title = self.fonts['title'].render(f"Selecione Composição - {team_name}", True, team_color)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 80))
        
        subtitle = self.fonts['medium'].render(f"Batalha {self.team_size}v{self.team_size}", True, Colors.GRAY)
        self.screen.blit(subtitle, (self.width//2 - subtitle.get_width()//2, 140))
        
        # Botões de composição
        if 'compositions' in self.buttons:
            for btn, config, name in self.buttons['compositions']:
                btn.color = Colors.BLUE_TEAM_DARK if self.selecting_team == "blue" else Colors.RED_TEAM_DARK
                btn.draw(self.screen, self.fonts['medium'])
                
                # Mostrar membros da composição
                members_text = " | ".join([f"{m['class'].title()}" for m in config])
                members_surf = self.fonts['small'].render(members_text, True, Colors.GRAY)
                self.screen.blit(members_surf, (btn.rect.x + 10, btn.rect.bottom - 25))
        
        # Instruções
        instructions = self.fonts['small'].render("ESC para voltar", True, Colors.DARK_GRAY)
        self.screen.blit(instructions, (20, self.height - 40))
    
    def _draw_battle(self):
        """Desenha a batalha"""
        if not self.battle_env:
            return
        
        # Área da arena (escalar para caber na tela)
        arena_w = self.battle_env.arena_config.width
        arena_h = self.battle_env.arena_config.height
        
        # Calcular escala para caber na tela com margem
        margin = 100
        scale_x = (self.width - margin * 2) / arena_w
        scale_y = (self.height - margin * 2 - 80) / arena_h  # 80 para UI
        scale = min(scale_x, scale_y)
        
        # Offset para centralizar
        offset_x = (self.width - arena_w * scale) / 2
        offset_y = 80 + (self.height - 80 - arena_h * scale) / 2
        
        # Desenhar arena
        arena_rect = pygame.Rect(
            offset_x, offset_y,
            arena_w * scale, arena_h * scale
        )
        pygame.draw.rect(self.screen, (40, 40, 50), arena_rect)
        pygame.draw.rect(self.screen, Colors.GRAY, arena_rect, 2)
        
        # Linha central
        pygame.draw.line(
            self.screen, (60, 60, 70),
            (self.width // 2, offset_y),
            (self.width // 2, offset_y + arena_h * scale),
            2
        )
        
        # Desenhar entidades
        for member in self.battle_env.blue_team:
            if member.entity.is_alive():
                self._draw_entity(member.entity, Colors.BLUE_TEAM, scale, offset_x, offset_y)
        
        for member in self.battle_env.red_team:
            if member.entity.is_alive():
                self._draw_entity(member.entity, Colors.RED_TEAM, scale, offset_x, offset_y)
        
        # UI superior
        self._draw_battle_ui()
    
    def _draw_entity(self, entity: Entity, color: Tuple[int, int, int], 
                     scale: float, offset_x: float, offset_y: float):
        """Desenha uma entidade na batalha"""
        x = offset_x + entity.x * scale
        y = offset_y + entity.y * scale
        radius = int(entity.radius * scale)
        
        # Corpo
        pygame.draw.circle(self.screen, color, (int(x), int(y)), radius)
        pygame.draw.circle(self.screen, Colors.WHITE, (int(x), int(y)), radius, 2)
        
        # Barra de vida
        bar_width = radius * 2
        bar_height = 4
        health_pct = entity.health / entity.stats_manager.get_stats().max_health
        
        bar_x = x - radius
        bar_y = y - radius - 10
        
        pygame.draw.rect(self.screen, Colors.DARK_GRAY, 
                        (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.screen, Colors.GREEN,
                        (bar_x, bar_y, bar_width * health_pct, bar_height))
    
    def _draw_battle_ui(self):
        """Desenha UI da batalha"""
        # Placar
        blue_alive = sum(1 for m in self.battle_env.blue_team if m.entity.is_alive())
        red_alive = sum(1 for m in self.battle_env.red_team if m.entity.is_alive())
        
        # Time Azul
        blue_text = self.fonts['large'].render(f"AZUL: {blue_alive}", True, Colors.BLUE_TEAM)
        self.screen.blit(blue_text, (50, 20))
        
        # Time Vermelho
        red_text = self.fonts['large'].render(f"VERMELHO: {red_alive}", True, Colors.RED_TEAM)
        self.screen.blit(red_text, (self.width - red_text.get_width() - 50, 20))
        
        # Info central
        step_text = self.fonts['medium'].render(
            f"Step: {self.battle_env.steps} | Velocidade: {self.battle_speed}x", 
            True, Colors.WHITE
        )
        self.screen.blit(step_text, (self.width//2 - step_text.get_width()//2, 25))
        
        # Status
        if not self.battle_running:
            if blue_alive > red_alive:
                status = "VITÓRIA AZUL!"
                status_color = Colors.BLUE_TEAM
            elif red_alive > blue_alive:
                status = "VITÓRIA VERMELHA!"
                status_color = Colors.RED_TEAM
            else:
                status = "EMPATE!"
                status_color = Colors.YELLOW
            
            status_surf = self.fonts['title'].render(status, True, status_color)
            self.screen.blit(status_surf, (self.width//2 - status_surf.get_width()//2, self.height//2))
            
            hint = self.fonts['medium'].render("ESPAÇO para reiniciar | ESC para sair", True, Colors.GRAY)
            self.screen.blit(hint, (self.width//2 - hint.get_width()//2, self.height//2 + 60))
        else:
            hint = self.fonts['small'].render(
                "ESPAÇO: pausar | 1/2/4: velocidade | ESC: sair", 
                True, Colors.DARK_GRAY
            )
            self.screen.blit(hint, (self.width//2 - hint.get_width()//2, self.height - 30))
    
    def _draw_stats(self):
        """Desenha tela de estatísticas"""
        title = self.fonts['title'].render("📊 Estatísticas da Sessão", True, Colors.WHITE)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 80))
        
        y = 200
        spacing = 50
        
        stats = [
            f"Batalhas jogadas: {self.session_stats['battles_played']}",
            f"Vitórias Azul: {self.session_stats['blue_wins']}",
            f"Vitórias Vermelho: {self.session_stats['red_wins']}",
            f"Empates: {self.session_stats['draws']}",
        ]
        
        for stat in stats:
            text = self.fonts['large'].render(stat, True, Colors.WHITE)
            self.screen.blit(text, (self.width//2 - text.get_width()//2, y))
            y += spacing
        
        # Taxa de vitória
        if self.session_stats['battles_played'] > 0:
            blue_rate = self.session_stats['blue_wins'] / self.session_stats['battles_played'] * 100
            red_rate = self.session_stats['red_wins'] / self.session_stats['battles_played'] * 100
            
            y += 30
            rate_text = f"Taxa Azul: {blue_rate:.1f}% | Taxa Vermelho: {red_rate:.1f}%"
            text = self.fonts['medium'].render(rate_text, True, Colors.GRAY)
            self.screen.blit(text, (self.width//2 - text.get_width()//2, y))
        
        # Instrução
        back = self.fonts['small'].render("ESC para voltar", True, Colors.DARK_GRAY)
        self.screen.blit(back, (20, self.height - 40))
    
    def _draw_tournament_setup(self):
        """Desenha configuração do torneio"""
        title = self.fonts['title'].render("🏆 Configuração do Torneio", True, Colors.ROLE_SUPPORT)
        self.screen.blit(title, (self.width//2 - title.get_width()//2, 80))
        
        info = self.fonts['medium'].render("Em desenvolvimento...", True, Colors.GRAY)
        self.screen.blit(info, (self.width//2 - info.get_width()//2, self.height//2))
        
        back = self.fonts['small'].render("ESC para voltar", True, Colors.DARK_GRAY)
        self.screen.blit(back, (20, self.height - 40))


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================

def run_tournament_ui():
    """Executa a interface do torneio"""
    # Obter tamanho do monitor
    pygame.init()
    info = pygame.display.Info()
    width = int(info.current_w * 0.85)
    height = int(info.current_h * 0.85)
    pygame.quit()
    
    # Criar e executar UI
    ui = TournamentUI(width, height)
    ui.run()


if __name__ == "__main__":
    run_tournament_ui()
