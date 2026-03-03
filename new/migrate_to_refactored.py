#!/usr/bin/env python3
"""
Migration script to help transition to refactored codebase
"""
import shutil
from pathlib import Path
import sys

def backup_original_files():
    """Backup original files"""
    files_to_backup = [
        "main.py",
        "enhanced_vni_classes.py",
        "smart_activation_router.py",
    ]
    
    for file_name in files_to_backup:
        file_path = Path(file_name)
        if file_path.exists():
            backup_path = Path(f"{file_name}.backup")
            shutil.copy2(file_path, backup_path)
            print(f"✅ Backed up {file_name} to {backup_path}")

def create_symlinks():
    """Create symlinks for backward compatibility"""
    try:
        # Create symlink for main.py
        target = Path("babybionn/main.py")
        link = Path("main_refactored.py")
        
        if link.exists() or link.is_symlink():
            link.unlink()
        
        # Create relative symlink
        link.symlink_to(target)
        print(f"✅ Created symlink {link} -> {target}")
        
    except Exception as e:
        print(f"❌ Failed to create symlinks: {e}")
        return False
    
    return True

def update_docker_compose():
    """Update Docker Compose file to use new entry point"""
    docker_compose_path = Path("docker-compose.dev.yml")
    if docker_compose_path.exists():
        try:
            with open(docker_compose_path, 'r') as f:
                content = f.read()
            
            # Replace main.py with main_refactored.py
            if "main.py" in content:
                content = content.replace("main.py", "main_refactored.py")
                with open(docker_compose_path, 'w') as f:
                    f.write(content)
                print("✅ Updated Docker Compose file")
                
        except Exception as e:
            print(f"❌ Failed to update Docker Compose: {e}")

def main():
    """Run migration steps"""
    print("🔄 Starting migration to refactored codebase...")
    print("=" * 60)
    
    # Step 1: Backup original files
    print("\n1. Backing up original files...")
    backup_original_files()
    
    # Step 2: Create symlinks
    print("\n2. Creating symlinks for backward compatibility...")
    if not create_symlinks():
        print("⚠️ Some symlinks could not be created")
    
    # Step 3: Update Docker Compose
    print("\n3. Updating Docker Compose configuration...")
    update_docker_compose()
    
    # Step 4: Instructions
    print("\n" + "=" * 60)
    print("✅ Migration completed!")
    print("\nNext steps:")
    print("1. Install the refactored package:")
    print("   pip install -e .")
    print("\n2. Test the new entry point:")
    print("   python main_refactored.py")
    print("\n3. Update your imports if needed:")
    print("   Old: from main import orchestrator")
    print("   New: from main_refactored import orchestrator")
    print("\n4. The original main.py is backed up as main.py.backup")
    print("\n5. To run with Docker:")
    print("   docker-compose -f docker-compose.dev.yml up --build")
    print("\nNote: The chat interface should work exactly as before!")

if __name__ == "__main__":
    main() 
