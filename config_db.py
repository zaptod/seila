"""
Banco de Dados Central de Configurações
=======================================
Sistema centralizado para armazenar e recuperar configurações
de classes e armas. Todas as features do jogo devem usar este
módulo para obter os parâmetros atualizados.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from stats import BaseStats, WeaponStats


# Arquivo de persistência
CONFIG_FILE = "custom_config.json"


class ConfigDatabase:
    """
    Banco de dados central de configurações.
    Singleton que gerencia todos os atributos customizados de classes e armas.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        
        # Configurações customizadas
        self._class_stats: Dict[str, Dict[str, float]] = {}
        self._weapon_stats: Dict[str, Dict[str, float]] = {}
        
        # Valores padrão originais (para reset)
        self._default_class_stats: Dict[str, Dict[str, float]] = {}
        self._default_weapon_stats: Dict[str, Dict[str, float]] = {}
        
        # Carregar configurações salvas
        self._load_from_file()
    
    def initialize_defaults(self, classes: Dict, weapons: Dict):
        """
        Inicializa os valores padrão a partir das classes e armas registradas.
        Deve ser chamado uma vez no início do jogo.
        """
        from entities import ClassRegistry
        from weapons import WeaponRegistry
        
        # Inicializar stats das classes
        for class_id in classes:
            cls = ClassRegistry.get_all().get(class_id)
            if cls:
                # Criar instância temporária para pegar stats
                temp = cls(0, 0, (255, 255, 255))
                base_stats = temp.stats_manager.get_stats()
                
                default_stats = {
                    'max_health': base_stats.max_health,
                    'speed': base_stats.speed,
                    'acceleration': base_stats.acceleration,
                    'defense': base_stats.defense,
                    'damage_multiplier': base_stats.damage_multiplier
                }
                
                self._default_class_stats[class_id] = default_stats.copy()
                
                # Só definir se não existir customização
                if class_id not in self._class_stats:
                    self._class_stats[class_id] = default_stats.copy()
        
        # Inicializar stats das armas
        for weapon_id in weapons:
            weapon_cls = WeaponRegistry.get_all().get(weapon_id)
            if weapon_cls:
                # Criar instância temporária
                class FakeOwner:
                    def __init__(self):
                        self.stats_manager = type('obj', (object,), {
                            'get_stats': lambda s: type('obj', (object,), {'attack_speed': 1.0})()
                        })()
                
                temp_weapon = weapon_cls(FakeOwner())
                stats = temp_weapon.stats
                
                default_stats = {
                    'base_damage': stats.base_damage,
                    'range': stats.range,
                    'attack_cooldown': stats.attack_cooldown,
                    'knockback_force': stats.knockback_force,
                    'critical_chance': stats.critical_chance
                }
                
                self._default_weapon_stats[weapon_id] = default_stats.copy()
                
                # Só definir se não existir customização
                if weapon_id not in self._weapon_stats:
                    self._weapon_stats[weapon_id] = default_stats.copy()
        
        # Salvar após inicialização
        self._save_to_file()
    
    # ========== GETTERS ==========
    
    def get_class_stats(self, class_id: str) -> Dict[str, float]:
        """Retorna os stats customizados de uma classe"""
        return self._class_stats.get(class_id, {}).copy()
    
    def get_weapon_stats(self, weapon_id: str) -> Dict[str, float]:
        """Retorna os stats customizados de uma arma"""
        return self._weapon_stats.get(weapon_id, {}).copy()
    
    def get_class_stat(self, class_id: str, stat_name: str) -> Optional[float]:
        """Retorna um stat específico de uma classe"""
        stats = self._class_stats.get(class_id, {})
        return stats.get(stat_name)
    
    def get_weapon_stat(self, weapon_id: str, stat_name: str) -> Optional[float]:
        """Retorna um stat específico de uma arma"""
        stats = self._weapon_stats.get(weapon_id, {})
        return stats.get(stat_name)
    
    def get_all_class_stats(self) -> Dict[str, Dict[str, float]]:
        """Retorna todos os stats de todas as classes"""
        return {k: v.copy() for k, v in self._class_stats.items()}
    
    def get_all_weapon_stats(self) -> Dict[str, Dict[str, float]]:
        """Retorna todos os stats de todas as armas"""
        return {k: v.copy() for k, v in self._weapon_stats.items()}
    
    def get_default_class_stats(self, class_id: str) -> Dict[str, float]:
        """Retorna os stats padrão originais de uma classe"""
        return self._default_class_stats.get(class_id, {}).copy()
    
    def get_default_weapon_stats(self, weapon_id: str) -> Dict[str, float]:
        """Retorna os stats padrão originais de uma arma"""
        return self._default_weapon_stats.get(weapon_id, {}).copy()
    
    # ========== SETTERS ==========
    
    def set_class_stat(self, class_id: str, stat_name: str, value: float):
        """Define um stat específico de uma classe"""
        if class_id not in self._class_stats:
            self._class_stats[class_id] = {}
        self._class_stats[class_id][stat_name] = value
        self._save_to_file()
    
    def set_weapon_stat(self, weapon_id: str, stat_name: str, value: float):
        """Define um stat específico de uma arma"""
        if weapon_id not in self._weapon_stats:
            self._weapon_stats[weapon_id] = {}
        self._weapon_stats[weapon_id][stat_name] = value
        self._save_to_file()
    
    def set_class_stats(self, class_id: str, stats: Dict[str, float]):
        """Define todos os stats de uma classe"""
        self._class_stats[class_id] = stats.copy()
        self._save_to_file()
    
    def set_weapon_stats(self, weapon_id: str, stats: Dict[str, float]):
        """Define todos os stats de uma arma"""
        self._weapon_stats[weapon_id] = stats.copy()
        self._save_to_file()
    
    # ========== RESET ==========
    
    def reset_class_stats(self, class_id: str):
        """Reseta os stats de uma classe para os valores padrão"""
        if class_id in self._default_class_stats:
            self._class_stats[class_id] = self._default_class_stats[class_id].copy()
            self._save_to_file()
    
    def reset_weapon_stats(self, weapon_id: str):
        """Reseta os stats de uma arma para os valores padrão"""
        if weapon_id in self._default_weapon_stats:
            self._weapon_stats[weapon_id] = self._default_weapon_stats[weapon_id].copy()
            self._save_to_file()
    
    def reset_all(self):
        """Reseta todas as configurações para os valores padrão"""
        self._class_stats = {k: v.copy() for k, v in self._default_class_stats.items()}
        self._weapon_stats = {k: v.copy() for k, v in self._default_weapon_stats.items()}
        self._save_to_file()
    
    # ========== PERSISTÊNCIA ==========
    
    def _save_to_file(self):
        """Salva as configurações em arquivo JSON"""
        try:
            data = {
                'classes': self._class_stats,
                'weapons': self._weapon_stats
            }
            with open(CONFIG_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Erro ao salvar configurações: {e}")
    
    def _load_from_file(self):
        """Carrega as configurações do arquivo JSON"""
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    self._class_stats = data.get('classes', {})
                    self._weapon_stats = data.get('weapons', {})
        except Exception as e:
            print(f"Erro ao carregar configurações: {e}")
            self._class_stats = {}
            self._weapon_stats = {}


# Instância global singleton
config_db = ConfigDatabase()


def get_config_db() -> ConfigDatabase:
    """Retorna a instância do banco de dados de configurações"""
    return config_db
