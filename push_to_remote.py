#!/usr/bin/env python3
"""
🚀 Remote Repository Update Script

This script helps you commit and push all the new repository setup files
to your remote Git repository.
"""

import subprocess
import sys
from pathlib import Path

# ANSI color codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

def print_step(text):
    print(f"{Colors.OKBLUE}🔧 {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.OKGREEN}✅ {text}{Colors.ENDC}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️  {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.ENDC}")

def run_command(cmd, check=True):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(cmd, shell=True, check=check, 
                              capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed: {cmd}")
        print_error(f"Error: {e.stderr}")
        return None, e.stderr

def check_git_status():
    """Check if we're in a git repository."""
    print_step("Checking Git repository status...")
    
    stdout, stderr = run_command("git status --porcelain", check=False)
    if stdout is None:
        print_error("Not in a Git repository or Git not available")
        return False
    
    print_success("Git repository detected")
    return True

def show_new_files():
    """Show the new files that were created."""
    print_step("New files added to your repository:")
    
    new_files = [
        ".github/workflows/ci.yml",
        ".github/pull_request_template.md", 
        ".github/ISSUE_TEMPLATE/bug_report.md",
        ".github/ISSUE_TEMPLATE/feature_request.md",
        ".github/ISSUE_TEMPLATE/question.md",
        "CONTRIBUTING.md",
        "LICENSE",
        "SECURITY.md",
        "CHANGELOG.md",
        "setup.py",
        "Dockerfile",
        "docker-compose.yml",
        "requirements-dev.txt",
        ".gitignore",
        "docs/REMOTE_SETUP.md"
    ]
    
    existing_files = []
    for file_path in new_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"   {Colors.OKCYAN}✓ {file_path}{Colors.ENDC}")
    
    if not existing_files:
        print_warning("No new files found to commit")
        return False
    
    return True

def git_add_commit():
    """Add and commit the new files."""
    print_step("Adding files to Git...")
    
    stdout, stderr = run_command("git add .")
    if stdout is None:
        print_error("Failed to add files to Git")
        return False
    
    print_success("Files added to Git staging area")
    
    print_step("Committing changes...")
    commit_message = """🚀 Add comprehensive remote repository setup

- Add GitHub Actions CI/CD pipeline
- Add issue and PR templates
- Add contributing guidelines and security policy
- Add Docker support with health checks
- Add automated setup script
- Add comprehensive documentation
- Add development dependencies
- Update README with badges and quick start
- Add changelog and license"""
    
    stdout, stderr = run_command(f'git commit -m "{commit_message}"')
    if stdout is None:
        if "nothing to commit" in stderr:
            print_warning("No changes to commit")
            return True
        print_error("Failed to commit changes")
        return False
    
    print_success("Changes committed successfully")
    return True

def git_push():
    """Push changes to remote repository."""
    print_step("Pushing changes to remote repository...")
    
    # Get current branch name
    stdout, stderr = run_command("git branch --show-current")
    if stdout is None:
        print_error("Failed to get current branch name")
        return False
    
    current_branch = stdout.strip()
    print(f"   Current branch: {Colors.OKCYAN}{current_branch}{Colors.ENDC}")
    
    # Push to remote
    stdout, stderr = run_command(f"git push origin {current_branch}")
    if stdout is None:
        print_error("Failed to push to remote repository")
        print_error("Make sure you have push access and the remote is configured")
        return False
    
    print_success("Changes pushed to remote repository successfully")
    return True

def show_next_steps():
    """Show next steps after successful push."""
    print_header("🎉 Repository Setup Complete!")
    
    print(f"{Colors.OKGREEN}Your remote repository is now fully configured!{Colors.ENDC}\n")
    
    print(f"{Colors.OKBLUE}📋 What was added:{Colors.ENDC}")
    print(f"   ✅ GitHub Actions CI/CD pipeline")
    print(f"   ✅ Issue and pull request templates")
    print(f"   ✅ Contributing guidelines")
    print(f"   ✅ Security policy and license")
    print(f"   ✅ Docker containerization support")
    print(f"   ✅ Automated setup script")
    print(f"   ✅ Comprehensive documentation")
    
    print(f"\n{Colors.OKBLUE}📱 GitHub Features Now Available:{Colors.ENDC}")
    print(f"   🔄 Automatic testing on push/PR")
    print(f"   📝 Professional issue templates")
    print(f"   🔍 Code security scanning")
    print(f"   📊 Project insights and analytics")
    print(f"   🏷️  Release management")
    print(f"   👥 Collaboration tools")
    
    print(f"\n{Colors.OKBLUE}🚀 Next Steps:{Colors.ENDC}")
    print(f"   1. Visit your repository on GitHub")
    print(f"   2. Enable branch protection rules")
    print(f"   3. Configure repository settings")
    print(f"   4. Invite collaborators")
    print(f"   5. Create your first release")
    
    print(f"\n{Colors.OKBLUE}📚 Documentation:{Colors.ENDC}")
    print(f"   📖 {Colors.OKCYAN}docs/REMOTE_SETUP.md{Colors.ENDC} - Complete setup guide")
    print(f"   🤝 {Colors.OKCYAN}CONTRIBUTING.md{Colors.ENDC} - Contribution guidelines")
    print(f"   🔒 {Colors.OKCYAN}SECURITY.md{Colors.ENDC} - Security policy")
    
    print(f"\n{Colors.WARNING}💡 Pro Tips:{Colors.ENDC}")
    print(f"   • Set up branch protection on main branch")
    print(f"   • Enable vulnerability alerts in repository settings")
    print(f"   • Configure GitHub Pages for documentation")
    print(f"   • Add repository description and topics")

def main():
    """Main function to update remote repository."""
    print_header("🏭 LLM Factory - Remote Repository Setup")
    print(f"{Colors.OKBLUE}This script will commit and push your repository setup files.{Colors.ENDC}\n")
    
    # Check if we're in a git repository
    if not check_git_status():
        sys.exit(1)
    
    # Show what files will be committed
    if not show_new_files():
        print(f"\n{Colors.WARNING}No new files to commit. Repository might already be up to date.{Colors.ENDC}")
        sys.exit(0)
    
    # Ask for confirmation
    print(f"\n{Colors.OKBLUE}Proceed with committing and pushing these files? (y/N): {Colors.ENDC}", end="")
    response = input().strip().lower()
    
    if response not in ['y', 'yes']:
        print(f"{Colors.WARNING}Operation cancelled by user{Colors.ENDC}")
        sys.exit(0)
    
    # Add and commit files
    if not git_add_commit():
        sys.exit(1)
    
    # Push to remote
    if not git_push():
        print(f"\n{Colors.WARNING}Push failed, but files are committed locally.{Colors.ENDC}")
        print(f"{Colors.WARNING}You can push manually later with: git push{Colors.ENDC}")
        sys.exit(1)
    
    # Show success message and next steps
    show_next_steps()
    
    print(f"\n{Colors.OKGREEN}🚀 Your LLM Factory repository is now production-ready!{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.WARNING}Operation interrupted by user{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        sys.exit(1)
