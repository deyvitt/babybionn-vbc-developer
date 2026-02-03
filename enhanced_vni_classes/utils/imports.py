# enhanced_vni_classes/utils/imports.py
"""
Dependency management and import utilities for VNI system.
Handles optional dependencies and provides fallbacks.
"""

import importlib
import sys
from typing import Optional, Any, Dict, List
from dataclasses import dataclass

@dataclass
class Dependency:
    """Represents a dependency with fallback options."""
    name: str
    required: bool = False
    fallback: Optional[str] = None
    install_name: Optional[str] = None
    purpose: str = ""
    
    @property
    def installed(self) -> bool:
        """Check if dependency is installed."""
        try:
            importlib.import_module(self.name if '.' not in self.name else self.name.split('.')[0])
            return True
        except ImportError:
            if self.fallback:
                try:
                    importlib.import_module(self.fallback)
                    return True
                except ImportError:
                    pass
            return False

class DependencyManager:
    """Manages dependencies for VNI system."""
    
    # Core dependencies (always required)
    CORE_DEPENDENCIES = [
        Dependency("json", required=True, purpose="JSON serialization"),
        Dependency("typing", required=True, purpose="Type hints"),
        Dependency("dataclasses", required=True, purpose="Data classes"),
        Dependency("enum", required=True, purpose="Enumerations"),
        Dependency("datetime", required=True, purpose="Date and time operations"),
        Dependency("hashlib", required=True, purpose="Hashing functions"),
        Dependency("collections", required=True, purpose="Collection types"),
        Dependency("re", required=True, purpose="Regular expressions"),
        Dependency("os", required=True, purpose="Operating system interfaces"),
        Dependency("sys", required=True, purpose="System-specific parameters"),
    ]
    
    # Optional dependencies with fallbacks
    OPTIONAL_DEPENDENCIES = [
        Dependency(
            name="aiohttp",
            required=False,
            install_name="aiohttp",
            purpose="Async HTTP client for web search"
        ),
        Dependency(
            name="numpy",
            required=False,
            fallback="math",
            install_name="numpy",
            purpose="Numerical computations for attention mechanisms"
        ),
        Dependency(
            name="yaml",
            required=False,
            fallback="json",
            install_name="PyYAML",
            purpose="YAML configuration parsing"
        ),
        Dependency(
            name="pickle",
            required=False,
            fallback="json",
            purpose="Object serialization for learning system"
        ),
        Dependency(
            name="asyncio",
            required=False,
            fallback=None,
            purpose="Asynchronous operations"
        ),
    ]
    
    def __init__(self):
        self.available_deps = {}
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check availability of all dependencies."""
        all_deps = self.CORE_DEPENDENCIES + self.OPTIONAL_DEPENDENCIES
        
        for dep in all_deps:
            self.available_deps[dep.name] = dep.installed
            
            if dep.required and not dep.installed:
                raise ImportError(
                    f"Required dependency '{dep.name}' is not installed. "
                    f"Purpose: {dep.purpose}. "
                    f"Install with: pip install {dep.install_name or dep.name}"
                )
    
    def is_available(self, module_name: str) -> bool:
        """Check if a module is available."""
        return self.available_deps.get(module_name, False)
    
    def get_import_warning(self, module_name: str) -> Optional[str]:
        """Get warning message for missing optional dependencies."""
        for dep in self.OPTIONAL_DEPENDENCIES:
            if dep.name == module_name and not dep.installed:
                if dep.fallback:
                    return (
                        f"Optional dependency '{module_name}' is not installed. "
                        f"Falling back to '{dep.fallback}'. "
                        f"For full functionality: pip install {dep.install_name or module_name}"
                    )
                else:
                    return (
                        f"Optional dependency '{module_name}' is not installed. "
                        f"Some features may be limited. "
                        f"Install with: pip install {dep.install_name or module_name}"
                    )
        return None
    
    def safe_import(self, module_name: str, fallback_module: Optional[str] = None) -> Any:
        """Safely import a module with fallback."""
        try:
            module = importlib.import_module(module_name)
            
            # Show warning if this was an optional dependency
            warning = self.get_import_warning(module_name)
            if warning:
                import warnings
                warnings.warn(warning, ImportWarning)
                
            return module
        except ImportError as e:
            if fallback_module:
                try:
                    return importlib.import_module(fallback_module)
                except ImportError:
                    pass
            
            # If no fallback or fallback also fails
            if module_name in [dep.name for dep in self.CORE_DEPENDENCIES if dep.required]:
                raise e
            else:
                raise ImportError(
                    f"Module '{module_name}' not available and no fallback succeeded. "
                    f"Original error: {e}"
                )
    
    def get_installation_commands(self, features: List[str]) -> Dict[str, str]:
        """Get installation commands for specific features."""
        feature_dependencies = {
            "web_search": ["aiohttp"],
            "attention_mechanisms": ["numpy"],
            "config_yaml": ["yaml"],
            "learning_persistence": ["pickle"],
            "async_operations": ["asyncio"]
        }
        
        commands = {}
        for feature in features:
            if feature in feature_dependencies:
                deps = feature_dependencies[feature]
                install_deps = []
                
                for dep_name in deps:
                    for dep in self.OPTIONAL_DEPENDENCIES:
                        if dep.name == dep_name and not dep.installed:
                            install_deps.append(dep.install_name or dep.name)
                
                if install_deps:
                    commands[feature] = f"pip install {' '.join(install_deps)}"
        
        return commands
    
    def check_feature_availability(self, feature: str) -> Dict[str, Any]:
        """Check availability of a specific feature."""
        feature_requirements = {
            "full_web_search": {
                "dependencies": ["aiohttp", "asyncio"],
                "optional": True,
                "description": "Full web search functionality"
            },
            "advanced_attention": {
                "dependencies": ["numpy"],
                "optional": True,
                "description": "Advanced attention mechanisms"
            },
            "yaml_config": {
                "dependencies": ["yaml"],
                "optional": True,
                "description": "YAML configuration support"
            },
            "learning_persistence": {
                "dependencies": ["pickle"],
                "optional": True,
                "description": "Learning system persistence"
            }
        }
        
        if feature not in feature_requirements:
            return {
                "available": False,
                "reason": f"Unknown feature: {feature}",
                "dependencies": []
            }
        
        reqs = feature_requirements[feature]
        missing = []
        
        for dep_name in reqs["dependencies"]:
            if not self.is_available(dep_name):
                missing.append(dep_name)
        
        return {
            "available": len(missing) == 0,
            "missing_dependencies": missing,
            "optional": reqs["optional"],
            "description": reqs["description"],
            "install_commands": self.get_installation_commands([feature])
        }

# Global dependency manager instance
_dep_manager: Optional[DependencyManager] = None

def get_dependency_manager() -> DependencyManager:
    """Get or create global dependency manager."""
    global _dep_manager
    if _dep_manager is None:
        _dep_manager = DependencyManager()
    return _dep_manager

def check_dependencies() -> Dict[str, bool]:
    """Check all dependencies and return availability status."""
    dm = get_dependency_manager()
    return dm.available_deps.copy()

def import_optional(module_name: str, fallback: Optional[str] = None) -> Any:
    """Import optional dependency with fallback."""
    dm = get_dependency_manager()
    return dm.safe_import(module_name, fallback)

def require_feature(feature: str) -> bool:
    """Check if a feature is available, raising error if required."""
    dm = get_dependency_manager()
    check = dm.check_feature_availability(feature)
    
    if not check["available"] and not check["optional"]:
        raise ImportError(
            f"Required feature '{feature}' is not available. "
            f"Missing dependencies: {check['missing_dependencies']}. "
            f"Install with: {check.get('install_commands', {}).get(feature, 'Check dependency manager')}"
        )
    
    return check["available"] 
