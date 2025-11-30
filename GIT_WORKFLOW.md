# Git Workflow Guide

This guide explains how to work with branches and create pull requests for each task.

## Current Branch Structure

```
main (base branch)
├── task-1 (Data Collection and Preprocessing)
├── task-2 (Sentiment and Thematic Analysis)
├── task-3 (PostgreSQL Database Setup)
└── task-4 (Insights and Recommendations)
```

## Workflow for Each Task

### Task 1: Data Collection and Preprocessing

1. **Switch to task-1 branch:**
   ```bash
   git checkout task-1
   ```

2. **Make your changes:**
   - Work on scraping and preprocessing
   - Test your code
   - Update files as needed

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Task 1: [Description of what you did]"
   ```

4. **Push to remote:**
   ```bash
   git push origin task-1
   ```

5. **Create Pull Request:**
   - Go to GitHub repository
   - Click "Compare & pull request" when prompted
   - Or manually: Click "Pull requests" → "New pull request"
   - Select `task-1` → `main`
   - Add description: "Task 1: Data Collection and Preprocessing"
   - Request review if needed
   - Merge when ready

6. **After merging, update main locally:**
   ```bash
   git checkout main
   git pull origin main
   ```

### Task 2: Sentiment and Thematic Analysis

1. **Switch to task-2 branch:**
   ```bash
   git checkout task-2
   ```

2. **Make sure you're up to date with main:**
   ```bash
   git merge main  # or git rebase main
   ```

3. **Work on Task 2:**
   - Implement sentiment analysis
   - Implement thematic analysis
   - Test your code

4. **Commit and push:**
   ```bash
   git add .
   git commit -m "Task 2: [Description]"
   git push origin task-2
   ```

5. **Create PR:** `task-2` → `main`

### Task 3: PostgreSQL Database Setup

1. **Switch to task-3 branch:**
   ```bash
   git checkout task-3
   git merge main  # Get latest from main
   ```

2. **Work on Task 3:**
   - Set up database schema
   - Implement data insertion
   - Test database operations

3. **Commit and push:**
   ```bash
   git add .
   git commit -m "Task 3: [Description]"
   git push origin task-3
   ```

4. **Create PR:** `task-3` → `main`

### Task 4: Insights and Recommendations

1. **Switch to task-4 branch:**
   ```bash
   git checkout task-4
   git merge main  # Get latest from main
   ```

2. **Work on Task 4:**
   - Generate insights
   - Create visualizations
   - Write recommendations

3. **Commit and push:**
   ```bash
   git add .
   git commit -m "Task 4: [Description]"
   git push origin task-4
   ```

4. **Create PR:** `task-4` → `main`

## Quick Reference Commands

### Check current branch:
```bash
git branch
```

### Switch branches:
```bash
git checkout <branch-name>
```

### Create new branch from main:
```bash
git checkout main
git pull origin main
git checkout -b <new-branch-name>
```

### See what branch you're on:
```bash
git status
```

### View commit history:
```bash
git log --oneline --graph --all
```

### Update branch with latest from main:
```bash
git checkout <your-branch>
git merge main
```

## Best Practices

1. **Always start from main:** When starting a new task, make sure your branch is up to date with main
2. **Commit frequently:** Make small, logical commits with clear messages
3. **Test before committing:** Make sure your code works before committing
4. **Write clear commit messages:** Use format "Task X: [What you did]"
5. **One task per branch:** Don't mix tasks in the same branch
6. **Update main after merge:** Always pull latest main after merging a PR

## Example Workflow Session

```bash
# Start working on Task 1
git checkout task-1

# Make changes, test...
# ... edit files ...

# Commit your work
git add .
git commit -m "Task 1: Implement review scraping with error handling"

# Continue working...
# ... more edits ...

# Commit again
git add .
git commit -m "Task 1: Add data preprocessing and validation"

# Push to remote
git push origin task-1

# Create PR on GitHub, then after merge:
git checkout main
git pull origin main
```

## Troubleshooting

### Accidentally committed to main?
```bash
# Create a new branch from current state
git checkout -b task-X
# Reset main to previous commit
git checkout main
git reset --hard origin/main
```

### Need to update your branch with main?
```bash
git checkout task-X
git merge main
# Resolve conflicts if any
git push origin task-X
```

### Want to see differences between branches?
```bash
git diff main..task-1
```

## Current Status

All branches are created and ready:
- ✅ `main` - Base branch with initial setup
- ✅ `task-1` - Ready for Task 1 work
- ✅ `task-2` - Ready for Task 2 work
- ✅ `task-3` - Ready for Task 3 work
- ✅ `task-4` - Ready for Task 4 work

You're currently on `task-1` branch. Start working on Task 1!

